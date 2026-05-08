"""
CDC FluView Prediction Pipeline - Optimized for GitHub Actions
Automated flu forecasting with prediction-vs-actual comparison and email alerts
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
import io
import json
import logging

warnings.filterwarnings('ignore')
plt.switch_backend('Agg')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
CDC_API = "https://api.delphi.cmu.edu/epidata/fluview/"
STATE = os.getenv("STATE", "California")

# Prediction persistence — stored for next-run comparison
PREDICTION_FILE = "output/last_prediction.json"

# Deviation threshold for comparison emails (0.2 = 20 %)
DEVIATION_THRESHOLD = float(os.getenv("DEVIATION_THRESHOLD", "0.2"))

# State abbreviation mapping for Delphi API (requires 2-letter codes)
_STATE_ABBR_MAP = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "puerto rico": "pr", "american samoa": "as", "guam": "gu",
    "northern mariana islands": "mp", "u.s. virgin islands": "vi",
}
_STATE_ABBR_MAP.update({v: v for v in _STATE_ABBR_MAP.values()})  # passthrough


def _get_state_abbr(state_name):
    """Resolve state name/abbreviation to a 2-letter code for the Delphi API."""
    key = state_name.strip().lower()
    if key in _STATE_ABBR_MAP:
        return _STATE_ABBR_MAP[key]
    raise ValueError(f"Unknown state: {state_name}")


def _epiweek_to_date(epiweek):
    """Convert MMWR epiweek (YYYYWW) to the Sunday start date.

    MMWR weeks start on Sunday. The first epiweek of the year is the week
    that contains January 4 (i.e., the first week with ≥4 days in January).

    When the previous year has 53 MMWR weeks (occurs when that year starts
    on a Sunday, or on a Saturday in a leap year), the first epiweek of the
    current year is shifted forward by one week.
    """
    year = epiweek // 100
    week = epiweek % 100
    jan4 = datetime(year, 1, 4)
    # Days since preceding Sunday (Sunday=0 … Saturday=6)
    sunday_offset = (jan4.weekday() + 1) % 7
    sunday_of_week1 = jan4 - timedelta(days=sunday_offset)

    # A year has 53 MMWR weeks when Jan 1 is Sunday (non-leap) or Saturday (leap)
    prev_jan1 = datetime(year - 1, 1, 1)
    prev_is_leap = ((year - 1) % 4 == 0 and (year - 1) % 100 != 0) or ((year - 1) % 400 == 0)
    if prev_jan1.weekday() == 6 or (prev_is_leap and prev_jan1.weekday() == 5):
        sunday_of_week1 += timedelta(weeks=1)

    return sunday_of_week1 + timedelta(weeks=week - 1)

# Email config
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_FROM = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_RECIPIENTS", "").split(",")


# ===========================================================================
# Data fetching
# ===========================================================================

def fetch_cdc_data(state_abbr=None):
    """Fetch flu data from CMU Delphi API (fluview endpoint).

    Parameters
    ----------
    state_abbr : str or None
        Two-letter state abbreviation (e.g. 'ca').  When None the
        STATE environment variable + abbreviation map is used.

    Returns a DataFrame with columns:
        week_start, week, ilitotal, total_patients, percent_ili
    """
    if state_abbr is None:
        state_abbr = _get_state_abbr(STATE)
    else:
        state_abbr = state_abbr.lower()

    logger.info("Fetching flu data for %s from Delphi API...", state_abbr.upper())

    # Compute epiweek range: 8 years of history, 1 year forward buffer
    now = datetime.now()
    start_year = now.year - 8
    end_year = now.year + 1
    epiweeks = f"{start_year}01-{end_year}52"

    params = {
        "regions": state_abbr,
        "epiweeks": epiweeks,
    }

    try:
        response = requests.get(CDC_API, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("result") != 1:
            raise ValueError(
                f"Delphi API error: {data.get('message', 'unknown error')}"
            )

        records = data.get("epidata", [])

        if not records:
            raise ValueError(f"No data returned from Delphi API for {state_abbr.upper()}")

    except requests.RequestException as e:
        logger.error("Failed to fetch data from Delphi API: %s", e)
        raise
    except Exception:
        raise

    # Normalize field names to match the rest of the pipeline
    for r in records:
        r["ilitotal"]      = r.pop("num_ili")
        r["total_patients"] = r.pop("num_patients")
        r["percent_ili"]   = r.pop("ili")

    df = pd.DataFrame(records)

    # Map Delphi field name → name the rest of the pipeline expects
    df = df.rename(columns={"epiweek": "week"})

    # Compute week_start from epiweek if the API doesn't provide it
    if 'week_start' not in df.columns:
        df['week_start'] = df['week'].apply(_epiweek_to_date)

    # Keep only columns we actually use
    cols = ["week_start", "week", "ilitotal", "total_patients", "percent_ili"]
    df = df[cols]

    df["week_start"] = pd.to_datetime(df["week_start"])
    df["week"] = df["week"].astype(int)
    df["ilitotal"] = pd.to_numeric(df["ilitotal"], errors="coerce")
    df["total_patients"] = pd.to_numeric(df["total_patients"], errors="coerce")
    df["percent_ili"] = pd.to_numeric(df["percent_ili"], errors="coerce")

    df = df.sort_values("week_start").reset_index(drop=True)
    logger.info(
        "✓ Fetched %d weeks (%s to %s)",
        len(df),
        df["week_start"].min().date(),
        df["week_start"].max().date(),
    )
    return df


# ===========================================================================
# Feature engineering
# ===========================================================================

def create_features(df):
    """Create time-series features for model training.

    All lagged and rolling features use *shifted* values to prevent
    data leakage — feature at row i never contains information from
    the target at row i.
    """
    df = df.copy().sort_values('week_start').reset_index(drop=True)
    target = 'ilitotal'

    # Lagged features (1–8 weeks)
    for lag in range(1, 9):
        df[f'lag_{lag}'] = df[target].shift(lag)

    # Rolling statistics (shift-then-roll to avoid leakage)
    for window in [2, 4, 8, 12]:
        df[f'roll_mean_{window}'] = df[target].shift(1).rolling(window).mean()
        df[f'roll_std_{window}'] = df[target].shift(1).rolling(window).std()

    # Seasonal features
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
    df['lag_52'] = df[target].shift(52)

    # Trend
    df['trend_4w'] = df[target].shift(1) - df[target].shift(5)

    return df.dropna(), target


def compute_prediction_features(last_row, historical_df):
    """Compute feature vector for *one* prediction row efficiently.

    Avoids recomputing features over the full dataset (O(1) instead of
    O(n)) by extracting only the values needed for the next-week prediction
    from already-loaded historical data.
    """
    target = 'ilitotal'
    result = {}
    last_values = historical_df[target].values
    n = len(last_values)

    # Lagged features
    for lag in range(1, 9):
        result[f'lag_{lag}'] = last_values[-lag] if n >= lag else np.nan

    # Rolling statistics (exclude the most recent row to match training logic)
    for window in [2, 4, 8, 12]:
        window_slice = last_values[-(window + 1):-1]
        result[f'roll_mean_{window}'] = float(np.mean(window_slice)) if len(window_slice) >= window else np.nan
        result[f'roll_std_{window}'] = float(np.std(window_slice)) if len(window_slice) >= 2 else 0.0

    # Seasonal features — compute CDC/MMWR week for the prediction date
    last_week = historical_df['week'].iloc[-1]
    next_week = 1 if last_week >= 52 else last_week + 1
    result['week_sin'] = np.sin(2 * np.pi * next_week / 52)
    result['week_cos'] = np.cos(2 * np.pi * next_week / 52)

    # Year-ago value
    result['lag_52'] = last_values[-52] if n >= 52 else np.nan

    # Trend
    result['trend_4w'] = last_values[-1] - last_values[-5] if n >= 5 else np.nan

    return result


# ===========================================================================
# Model
# ===========================================================================

def build_model():
    """Build a LightGBM regressor with validated hyperparameters.

    num_leaves (31) ≤ 2^max_depth (64), avoiding the warning from the
    previous configuration where 50 leaves exceeded the depth-5 budget.
    """
    return lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )


def train_and_evaluate(df):
    """Train model with a chronological split and honest out-of-sample metrics.

    Splits the time-series 80/20 (earlier/later), reports both training
    and validation RMSE/R², then retrains on the full dataset for the
    final one-week-ahead forecast.
    """
    logger.info("Training model with time-series validation...")

    df_feat, target = create_features(df)

    feature_cols = [c for c in df_feat.columns
                    if c not in ['week_start', 'week', 'ilitotal',
                                 'total_patients', 'percent_ili']]

    X = df_feat[feature_cols].values
    y = df_feat[target].values

    # Chronological split: train on first 80%, validate on last 20%
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    model = build_model()
    model.fit(X_train, y_train)

    # --- Training metrics (in-sample, informational only) ---
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    # --- Validation metrics (out-of-sample — these matter) ---
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)

    logger.info("✓ Training   - RMSE: %.2f, R²: %.3f", train_rmse, train_r2)
    logger.info("✓ Validation - RMSE: %.2f, R²: %.3f", val_rmse, val_r2)

    # Retrain on full data for the final one-week-ahead prediction
    model.fit(X, y)

    return model, feature_cols, val_rmse, val_r2


# ===========================================================================
# Prediction
# ===========================================================================

def predict_future(model, df, feature_cols):
    """Generate next-week prediction with empirical prediction intervals.

    Uses compute_prediction_features for O(1) feature construction and
    non-parametric empirical quantiles of recent residuals to build a
    95 % prediction interval (no normality assumption).
    """
    logger.info("Predicting next week...")

    target = 'ilitotal'
    last_date = df['week_start'].max()
    next_date = last_date + timedelta(weeks=1)

    last_row = df.iloc[-1].copy()
    last_row['week_start'] = next_date

    # Compute features for the single prediction point
    feat = compute_prediction_features(last_row, df)
    X_pred = np.array([[feat[col] for col in feature_cols]])
    pred_value = max(0, model.predict(X_pred)[0])

    # --- Non-parametric 95 % prediction interval ---
    # Derive error distribution from the most recent 30 residuals
    df_feat, _ = create_features(df)
    recent = df_feat.tail(30)
    X_recent = recent[feature_cols].values
    recent_preds = model.predict(X_recent)
    residuals = recent[target].values - recent_preds

    lower_resid = np.percentile(residuals, 2.5)
    upper_resid = np.percentile(residuals, 97.5)

    ci_lower = max(0, pred_value + lower_resid)
    ci_upper = pred_value + upper_resid

    logger.info("  Predicted: %.0f  [%.0f, %.0f]", pred_value, ci_lower, ci_upper)

    return pd.DataFrame([{
        'week_start': next_date,
        'predicted_ili': pred_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }])


# ===========================================================================
# Prediction persistence (cross-run comparison)
# ===========================================================================

def save_prediction(prediction):
    """Persist the one-week-ahead forecast so the next run can compare it
    against actual data when it becomes available.

    Parameters
    ----------
    prediction : pd.DataFrame
        One-row DataFrame with columns week_start, predicted_ili,
        ci_lower, ci_upper.
    """
    os.makedirs('output', exist_ok=True)

    record = {
        'week_start': prediction['week_start'].iloc[0].strftime('%Y-%m-%d'),
        'predicted_ili': float(prediction['predicted_ili'].iloc[0]),
        'ci_lower': float(prediction['ci_lower'].iloc[0]),
        'ci_upper': float(prediction['ci_upper'].iloc[0]),
        'generated_at': datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }

    with open(PREDICTION_FILE, 'w') as f:
        json.dump(record, f, indent=2)

    logger.info("✓ Prediction saved to %s", PREDICTION_FILE)


def load_saved_prediction():
    """Load the forecast saved by the previous pipeline run.

    Returns
    -------
    dict or None
        Dictionary with keys week_start, predicted_ili, ci_lower,
        ci_upper, generated_at.  Returns None when no prior prediction
        exists or the file is unreadable.
    """
    if not os.path.exists(PREDICTION_FILE):
        logger.info("No saved prediction found — this is a cold start")
        return None

    try:
        with open(PREDICTION_FILE, 'r') as f:
            record = json.load(f)

        # Validate required keys
        for key in ('week_start', 'predicted_ili', 'ci_lower', 'ci_upper'):
            if key not in record:
                logger.warning("Saved prediction is missing key '%s' — discarding", key)
                return None

        logger.info("✓ Loaded saved prediction for week %s (predicted: %.0f)",
                    record['week_start'], record['predicted_ili'])
        return record

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("Failed to parse saved prediction: %s — treating as cold start", e)
        return None


# ===========================================================================
# Prediction-vs-actual comparison
# ===========================================================================

def find_actual_for_prediction(df, saved_pred):
    """Look up the actual ILI count in *df* for the week that was predicted.

    Parameters
    ----------
    df : pd.DataFrame
        Fetched CDC data with a 'week_start' column (datetime) and
        'ilitotal' column.
    saved_pred : dict
        Saved prediction with key 'week_start' (str in YYYY-MM-DD format).

    Returns
    -------
    float or None
        The actual ilitotal for the predicted week, or None when that
        week has not yet appeared in the CDC dataset.
    """
    pred_date = pd.Timestamp(saved_pred['week_start'])

    match = df[df['week_start'] == pred_date]

    if len(match) == 0:
        logger.info("Predicted week %s not yet in CDC data", saved_pred['week_start'])
        return None

    actual = match['ilitotal'].iloc[0]

    if pd.isna(actual):
        logger.info("Predicted week %s exists but ILI value is NaN", saved_pred['week_start'])
        return None

    logger.info("✓ Found actual ILI for %s: %.0f", saved_pred['week_start'], actual)
    return float(actual)


# ===========================================================================
# Visualisation
# ===========================================================================

def create_plot(historical, prediction, actual_point=None):
    """Create a forecast visualisation chart as a PNG buffer.

    Parameters
    ----------
    historical : pd.DataFrame
        Full historical CDC dataset.
    prediction : pd.DataFrame
        One-row DataFrame with next-week forecast.
    actual_point : dict or None
        When a prior prediction has been compared against actual data,
        pass {'week_start': str, 'actual_ili': float, 'predicted_ili': float}
        to place markers for both on the chart.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Historical data (last year)
    hist_recent = historical.tail(52)
    ax.plot(hist_recent['week_start'], hist_recent['ilitotal'],
            label='Historical ILI Cases', marker='o', linewidth=2,
            color='#2196F3')

    # Next-week prediction
    ax.plot(prediction['week_start'], prediction['predicted_ili'],
            label='Next Week Prediction', marker='s', markersize=12,
            linewidth=2.5, linestyle='--', color='#F44336')

    # Prediction interval as error bar
    ax.errorbar(
        prediction['week_start'].iloc[0],
        prediction['predicted_ili'].iloc[0],
        yerr=[[prediction['predicted_ili'].iloc[0] -
               prediction['ci_lower'].iloc[0]],
              [prediction['ci_upper'].iloc[0] -
               prediction['predicted_ili'].iloc[0]]],
        fmt='none', color='#F44336', capsize=10, capthick=2,
        label='95% Prediction Interval', alpha=0.6
    )

    # --- Comparison markers (when prior prediction has been evaluated) ---
    if actual_point is not None:
        pred_date = pd.Timestamp(actual_point['week_start'])
        pred_val = actual_point['predicted_ili']
        actual_val = actual_point['actual_ili']

        # Prior prediction marker
        ax.scatter(pred_date, pred_val,
                   color='#9C27B0', s=140, zorder=5, marker='D',
                   edgecolors='#4A148C', linewidths=1.5,
                   label=f'Prior Prediction ({pred_val:.0f})')

        # Actual value marker
        ax.scatter(pred_date, actual_val,
                   color='#4CAF50', s=160, zorder=6, marker='o',
                   edgecolors='#1B5E20', linewidths=2,
                   label=f'Actual ({actual_val:.0f})')

        # Connecting line between predicted and actual
        ax.plot([pred_date, pred_date], [min(pred_val, actual_val), max(pred_val, actual_val)],
                color='#666666', linewidth=1.5, linestyle=':', alpha=0.7)

    ax.set_xlabel('Week Starting', fontsize=12, fontweight='bold')
    ax.set_ylabel('ILI Cases', fontsize=12, fontweight='bold')
    ax.set_title(
        f'{STATE} Flu Activity — Next Week Forecast\n'
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return buf


# ===========================================================================
# Email
# ===========================================================================

def send_comparison_email(saved_pred, actual_value, new_prediction, plot_buf):
    """Send an HTML email comparing the prior prediction against actual data.

    Fires when the absolute percentage deviation between predicted and
    actual ILI exceeds DEVIATION_THRESHOLD.  Includes the new one-week-ahead
    forecast for reference.
    """
    if not EMAIL_FROM or not EMAIL_PASSWORD or not EMAIL_TO[0]:
        logger.warning("Email not configured — skipping comparison notification")
        return False

    predicted = saved_pred['predicted_ili']
    ci_lower = saved_pred['ci_lower']
    ci_upper = saved_pred['ci_upper']
    deviation_pct = abs(predicted - actual_value) / actual_value * 100
    in_interval = ci_lower <= actual_value <= ci_upper
    over_under = "over-predicted" if predicted > actual_value else "under-predicted"

    logger.info("Sending comparison email to %d recipient(s)...", len(EMAIL_TO))

    msg = MIMEMultipart('related')
    msg['Subject'] = (
        f'📊 Flu Prediction Accuracy — {STATE}: '
        f'{over_under} by {deviation_pct:.0f}%'
    )
    msg['From'] = EMAIL_FROM
    msg['To'] = ', '.join(EMAIL_TO)

    interval_status = (
        '<span style="color: #4CAF50; font-weight: bold;">✓ Yes — actual fell within the 95% PI</span>'
        if in_interval
        else '<span style="color: #F44336; font-weight: bold;">✗ No — actual was outside the 95% PI</span>'
    )

    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
            .deviation {{ background: #fce4ec; padding: 20px;
                         border-left: 5px solid #d32f2f; margin: 20px 0; }}
            .metric {{ font-size: 28px; font-weight: bold; color: #d32f2f; }}
            .good {{ color: #4CAF50; font-weight: bold; }}
            .bad {{ color: #d32f2f; font-weight: bold; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background: #f2f2f2; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h2>📊 Prediction Accuracy Report — {STATE}</h2>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>

        <div class="deviation">
            <h3>⚠️ Significant Deviation Detected</h3>
            <p>The model <strong>{over_under}</strong> ILI cases for week
            <strong>{saved_pred['week_start']}</strong> by
            <span class="metric">{deviation_pct:.1f}%</span></p>
            <p>Deviation threshold for alert: {DEVIATION_THRESHOLD*100:.0f}%</p>
        </div>

        <h3>📋 Comparison Summary:</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td><strong>Week</strong></td>
                <td>{saved_pred['week_start']}</td>
            </tr>
            <tr>
                <td><strong>Predicted ILI</strong></td>
                <td>{predicted:.0f} (95% PI: {ci_lower:.0f} – {ci_upper:.0f})</td>
            </tr>
            <tr>
                <td><strong>Actual ILI</strong></td>
                <td class="bad">{actual_value:.0f}</td>
            </tr>
            <tr>
                <td><strong>Absolute Deviation</strong></td>
                <td class="bad">{abs(predicted - actual_value):.0f} cases ({deviation_pct:.1f}%)</td>
            </tr>
            <tr>
                <td><strong>Within 95% PI?</strong></td>
                <td>{interval_status}</td>
            </tr>
            <tr>
                <td><strong>Direction</strong></td>
                <td>{over_under.replace('-', ' ').title()}</td>
            </tr>
        </table>

        <h3>🔮 New Forecast (Next Week):</h3>
        <table>
            <tr>
                <th>Week Starting</th>
                <th>Predicted ILI</th>
                <th>95% PI</th>
            </tr>
            <tr>
                <td><strong>{new_prediction['week_start'].iloc[0].strftime('%Y-%m-%d')}</strong></td>
                <td>{new_prediction['predicted_ili'].iloc[0]:.0f}</td>
                <td>{new_prediction['ci_lower'].iloc[0]:.0f} – {new_prediction['ci_upper'].iloc[0]:.0f}</td>
            </tr>
        </table>

        <h3>📈 Forecast Chart:</h3>
        <img src="cid:forecast_plot" alt="Flu Forecast Chart"
             style="max-width: 100%%; height: auto;
                    border: 1px solid #ddd; border-radius: 4px;">

        <p style="margin-top: 30px; color: #666; font-size: 12px;">
            <em>Automated comparison from the CDC Flu Prediction System (GitHub
            Actions)</em><br>
            Data source: CMU Delphi FluView API  |  State: {STATE}<br>
            Deviation threshold: {DEVIATION_THRESHOLD*100:.0f}%  |
            Model: LightGBM<br>
            Chart is also attached as flu_forecast.png
        </p>
    </body>
    </html>
    """

    msg.attach(MIMEText(html, 'html'))

    # Attach plot as inline image
    img = MIMEImage(plot_buf.getvalue())
    img.add_header('Content-ID', '<forecast_plot>')
    img.add_header('Content-Disposition', 'attachment', filename='flu_forecast.png')
    msg.attach(img)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info("✓ Comparison email sent successfully!")
        return True
    except Exception as e:
        logger.error("Failed to send comparison email: %s", e)
        return False


# ===========================================================================
# Artifacts
# ===========================================================================

def save_artifacts(prediction, plot_buf, comparison_made=False,
                   actual_point=None, saved_pred=None):
    """Persist prediction CSV, chart PNG, and text summary to output/.

    Parameters
    ----------
    prediction : pd.DataFrame
        One-row DataFrame with the new one-week-ahead forecast.
    plot_buf : io.BytesIO
        PNG buffer of the forecast chart.
    comparison_made : bool
        Whether prediction-vs-actual comparison was performed this run.
    actual_point : dict or None
        {'week_start': str, 'actual_ili': float, 'predicted_ili': float}
    saved_pred : dict or None
        The prior saved prediction that was compared.
    """
    try:
        os.makedirs('output', exist_ok=True)

        prediction.to_csv('output/prediction.csv', index=False)
        logger.info("✓ Prediction saved to output/prediction.csv")

        with open('output/forecast_plot.png', 'wb') as f:
            f.write(plot_buf.getvalue())
        logger.info("✓ Plot saved to output/forecast_plot.png")

        with open('output/summary.txt', 'w') as f:
            f.write("CDC Flu Prediction Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            f.write(f"State: {STATE}\n\n")

            # --- Comparison section ---
            if comparison_made and actual_point is not None and saved_pred is not None:
                deviation_pct = (
                    abs(saved_pred['predicted_ili'] - actual_point['actual_ili'])
                    / actual_point['actual_ili'] * 100
                )
                f.write("Prior Prediction Comparison:\n")
                f.write(f"  Week: {actual_point['week_start']}\n")
                f.write(f"  Predicted: {saved_pred['predicted_ili']:.0f}\n")
                f.write(f"  Actual:    {actual_point['actual_ili']:.0f}\n")
                f.write(f"  Deviation: {deviation_pct:.1f}%\n\n")

            f.write("Next Week Prediction:\n")
            f.write(f"  Week Starting: {prediction['week_start'].iloc[0].strftime('%Y-%m-%d')}\n")
            f.write(f"  Predicted ILI: {prediction['predicted_ili'].iloc[0]:.0f}\n")
            f.write(f"  95% PI: [{prediction['ci_lower'].iloc[0]:.0f}, "
                     f"{prediction['ci_upper'].iloc[0]:.0f}]\n")
        logger.info("✓ Summary saved to output/summary.txt")

        return True
    except Exception as e:
        logger.error("Failed to save artifacts: %s", e)
        return False


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_pipeline():
    """Execute the full CDC flu prediction pipeline.

    Flow
    ----
    1. Fetch latest CDC data.
    2. Load the prediction saved by the *previous* run.
    3. If a saved prediction exists:
       a. Look up actual ILI for that predicted week.
       b. If not yet available → exit cleanly (nothing to do).
       c. If available → compare, log, and send email if the deviation
          exceeds DEVIATION_THRESHOLD.
    4. Retrain the model on all available data.
    5. Predict one week ahead.
    6. Save the new prediction for the next run.
    7. Generate plot and artifacts.
    """
    logger.info("=" * 70)
    logger.info("CDC FLU PREDICTION PIPELINE — GITHUB ACTIONS")
    logger.info("=" * 70)

    try:
        # 1. Fetch data
        df = fetch_cdc_data()

        # 2. Load saved prediction from previous run
        saved_pred = load_saved_prediction()
        comparison_made = False
        actual_value = None

        # 3. If a saved prediction exists, check whether actual data is available
        if saved_pred is not None:
            actual_value = find_actual_for_prediction(df, saved_pred)

            if actual_value is None:
                # ---- No new data yet — exit cleanly ----
                logger.info("")
                logger.info("=" * 70)
                logger.info("NO NEW DATA YET — EXITING")
                logger.info("=" * 70)
                logger.info(
                    "Predicted week %s data not yet available from CDC.",
                    saved_pred['week_start'],
                )
                logger.info(
                    "Last prediction: %.0f  [%.0f, %.0f]",
                    saved_pred['predicted_ili'],
                    saved_pred['ci_lower'],
                    saved_pred['ci_upper'],
                )
                logger.info("Nothing to do — pipeline will retry on next scheduled run.")
                logger.info("=" * 70)
                return 0

            # ---- Data is available — compare ----
            predicted = saved_pred['predicted_ili']
            deviation_pct = abs(predicted - actual_value) / actual_value * 100
            in_interval = (
                saved_pred['ci_lower'] <= actual_value <= saved_pred['ci_upper']
            )

            logger.info("")
            logger.info("=" * 70)
            logger.info("PREDICTION vs ACTUAL COMPARISON")
            logger.info("=" * 70)
            logger.info("  Week:          %s", saved_pred['week_start'])
            logger.info("  Predicted:     %.0f  [%.0f, %.0f]",
                        predicted, saved_pred['ci_lower'], saved_pred['ci_upper'])
            logger.info("  Actual:        %.0f", actual_value)
            logger.info("  Deviation:     %.1f%%", deviation_pct)
            logger.info("  Within 95%% PI: %s", "Yes" if in_interval else "No")
            logger.info("=" * 70)

            comparison_made = True

        # 4. Train model on full dataset
        model, feature_cols, val_rmse, val_r2 = train_and_evaluate(df)

        # 5. Predict next week
        prediction = predict_future(model, df, feature_cols)

        # 6. Save the new prediction for the next run
        save_prediction(prediction)

        # Log the new prediction
        logger.info("")
        logger.info("=" * 70)
        logger.info("NEXT WEEK PREDICTION")
        logger.info("=" * 70)
        logger.info("  Week Starting:    %s",
                    prediction['week_start'].iloc[0].strftime('%Y-%m-%d'))
        logger.info("  Predicted ILI:    %.0f", prediction['predicted_ili'].iloc[0])
        logger.info("  95%% PI:           [%.0f, %.0f]",
                    prediction['ci_lower'].iloc[0],
                    prediction['ci_upper'].iloc[0])
        logger.info("  Validation RMSE:  %.2f", val_rmse)
        logger.info("=" * 70)

        # 7. Create visualization — include comparison markers when available
        actual_point = None
        if comparison_made and actual_value is not None:
            actual_point = {
                'week_start': saved_pred['week_start'],
                'actual_ili': actual_value,
                'predicted_ili': saved_pred['predicted_ili'],
            }
        plot_buf = create_plot(df, prediction, actual_point)

        # 8. Save artifacts
        save_artifacts(
            prediction, plot_buf,
            comparison_made=comparison_made,
            actual_point=actual_point,
            saved_pred=saved_pred,
        )

        # 9. Send comparison email if deviation exceeds threshold
        if comparison_made and actual_value is not None:
            deviation_pct = (
                abs(saved_pred['predicted_ili'] - actual_value)
                / actual_value * 100
            )
            if deviation_pct > DEVIATION_THRESHOLD * 100:
                logger.warning(
                    "⚠️  Prediction deviation %.1f%% exceeds threshold %.0f%%",
                    deviation_pct, DEVIATION_THRESHOLD * 100,
                )
                send_comparison_email(
                    saved_pred, actual_value, prediction, plot_buf,
                )

                # Write comparison output for GitHub Actions
                with open(os.environ.get('GITHUB_OUTPUT', '/dev/null'), 'a') as f:
                    f.write("comparison_alert=true\n")
                    f.write(f"comparison_week={saved_pred['week_start']}\n")
                    f.write(f"comparison_predicted={saved_pred['predicted_ili']:.0f}\n")
                    f.write(f"comparison_actual={actual_value:.0f}\n")
                    f.write(f"comparison_deviation_pct={deviation_pct:.1f}\n")
            else:
                logger.info(
                    "✓ Prediction deviation %.1f%% within acceptable range "
                    "(threshold: %.0f%%)",
                    deviation_pct, DEVIATION_THRESHOLD * 100,
                )
                with open(os.environ.get('GITHUB_OUTPUT', '/dev/null'), 'a') as f:
                    f.write("comparison_alert=false\n")
                    f.write(f"comparison_deviation_pct={deviation_pct:.1f}\n")

        logger.info("")
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error("")
        logger.error("=" * 70)
        logger.error("PIPELINE FAILED!")
        logger.error("=" * 70)
        logger.error("Error: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    exit(run_pipeline())
