"""
CDC FluView Prediction Pipeline - Optimized for GitHub Actions
Automated flu forecasting with email alerts
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
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
THRESHOLD = float(os.getenv("THRESHOLD_MULTIPLIER", "1.1"))

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

# Email config
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_FROM = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_RECIPIENTS", "").split(",")

# Alert deduplication — don't re-alert for the same week within N days
ALERT_COOLDOWN_DAYS = 6


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


def check_alerts(prediction, recent_avg, threshold_mult=1.1):
    """Determine whether the prediction crosses the alert threshold."""
    threshold = recent_avg * threshold_mult

    if prediction['predicted_ili'].iloc[0] > threshold:
        return {
            'week': prediction['week_start'].iloc[0],
            'predicted': prediction['predicted_ili'].iloc[0],
            'threshold': threshold,
            'excess': prediction['predicted_ili'].iloc[0] - threshold,
            'recent_avg': recent_avg
        }, threshold

    return None, threshold


# ===========================================================================
# Alert deduplication
# ===========================================================================

def should_send_alert(alert):
    """Enforce a cooldown so the same week doesn't trigger duplicate alerts.

    Writes the alert week to ``output/last_alert.txt`` and suppresses any
    alert within ``ALERT_COOLDOWN_DAYS`` of the last one.
    """
    cooldown_file = 'output/last_alert.txt'

    try:
        if os.path.exists(cooldown_file):
            with open(cooldown_file, 'r') as f:
                lines = f.read().strip().split('\n')
                if lines:
                    last_alert_date = datetime.strptime(lines[0], '%Y-%m-%d')
                    days_since = (datetime.now() - last_alert_date).days
                    if days_since < ALERT_COOLDOWN_DAYS:
                        logger.info(
                            "Alert suppressed: last alert %d day(s) ago "
                            "(cooldown: %d days)",
                            days_since, ALERT_COOLDOWN_DAYS
                        )
                        return False

        # Record this alert
        os.makedirs('output', exist_ok=True)
        with open(cooldown_file, 'w') as f:
            f.write(f"{alert['week'].strftime('%Y-%m-%d')}\n")
            f.write(f"predicted={alert['predicted']:.0f}\n")
            f.write(f"threshold={alert['threshold']:.0f}\n")
        return True

    except Exception as e:
        logger.warning("Alert cooldown check failed, sending anyway: %s", e)
        return True


# ===========================================================================
# Visualisation
# ===========================================================================

def create_plot(historical, prediction, threshold):
    """Create a forecast visualisation chart as a PNG buffer."""
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

    # Threshold line
    ax.axhline(y=threshold, color='#FF9800', linestyle=':',
               linewidth=2.5, label=f'Alert Threshold ({threshold:.0f})')

    # Star marker when threshold is exceeded
    if prediction['predicted_ili'].iloc[0] > threshold:
        ax.scatter(prediction['week_start'], prediction['predicted_ili'],
                   color='#F44336', s=300, zorder=5, marker='*',
                   edgecolors='darkred', linewidths=2,
                   label='⚠️ Threshold Exceeded')

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

def send_email(alert, prediction, plot_buf, threshold):
    """Send an HTML alert email with the forecast chart embedded inline."""
    if not EMAIL_FROM or not EMAIL_PASSWORD or not EMAIL_TO[0]:
        logger.warning("Email not configured — skipping notification")
        return False

    logger.info("Sending email to %d recipient(s)...", len(EMAIL_TO))

    msg = MIMEMultipart('related')
    msg['Subject'] = f'🚨 Flu Alert: Next week exceeds threshold — {STATE}'
    msg['From'] = EMAIL_FROM
    msg['To'] = ', '.join(EMAIL_TO)

    # HTML body — now includes the inline chart via cid:forecast_plot
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
            .alert {{ background: #fff3cd; padding: 20px;
                      border-left: 5px solid #ffc107; margin: 20px 0; }}
            .metric {{ font-size: 28px; font-weight: bold; color: #d32f2f; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background: #f2f2f2; font-weight: bold; }}
            .exceed {{ color: #d32f2f; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h2>🚨 Flu Activity Alert — {STATE}</h2>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>

        <div class="alert">
            <h3>⚠️ Threshold Exceeded for Next Week</h3>
            <p><span class="metric">{alert['predicted']:.0f}</span> predicted ILI cases</p>
            <p>This is <strong>{alert['excess']:.0f} cases</strong> above the alert threshold
            ({(alert['excess']/alert['threshold']*100):.1f}% increase)</p>
        </div>

        <h3>📊 Next Week Forecast:</h3>
        <table>
            <tr>
                <th>Week Starting</th>
                <th>Predicted ILI Cases</th>
                <th>95% Prediction Interval</th>
                <th>Alert Threshold</th>
                <th>Recent 8-Week Avg</th>
            </tr>
            <tr>
                <td><strong>{prediction['week_start'].iloc[0].strftime('%Y-%m-%d')}</strong></td>
                <td class="exceed">{prediction['predicted_ili'].iloc[0]:.0f}</td>
                <td>{prediction['ci_lower'].iloc[0]:.0f} – {prediction['ci_upper'].iloc[0]:.0f}</td>
                <td>{alert['threshold']:.0f}</td>
                <td>{alert['recent_avg']:.0f}</td>
            </tr>
        </table>

        <h3>📈 Key Metrics:</h3>
        <ul>
            <li><strong>Predicted Cases:</strong> {alert['predicted']:.0f}</li>
            <li><strong>Alert Threshold:</strong> {alert['threshold']:.0f}
                ({THRESHOLD*100:.0f}% of recent average)</li>
            <li><strong>Excess Cases:</strong> +{alert['excess']:.0f}
                ({(alert['excess']/alert['threshold']*100):.1f}%)</li>
            <li><strong>Prediction Interval:</strong>
                {prediction['ci_lower'].iloc[0]:.0f} – {prediction['ci_upper'].iloc[0]:.0f}</li>
            <li><strong>Recent 8-Week Average:</strong> {alert['recent_avg']:.0f}</li>
        </ul>

        <h3>📈 Forecast Chart:</h3>
        <img src="cid:forecast_plot" alt="Flu Forecast Chart"
             style="max-width: 100%%; height: auto;
                    border: 1px solid #ddd; border-radius: 4px;">

        <p style="margin-top: 30px; color: #666; font-size: 12px;">
            <em>Automated alert from the CDC Flu Prediction System (GitHub Actions)</em><br>
            Data source: CMU Delphi FluView API  |  State: {STATE}<br>
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
        logger.info("✓ Email sent successfully!")
        return True
    except Exception as e:
        logger.error("Failed to send email: %s", e)
        return False


# ===========================================================================
# Artifacts
# ===========================================================================

def save_artifacts(prediction, plot_buf):
    """Persist prediction CSV, chart PNG, and text summary to output/."""
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
    """Execute the full CDC flu prediction pipeline."""
    logger.info("=" * 70)
    logger.info("CDC FLU PREDICTION PIPELINE — GITHUB ACTIONS")
    logger.info("=" * 70)

    try:
        # 1. Fetch data
        df = fetch_cdc_data()

        # 2. Train + evaluate (chronological split)
        model, feature_cols, val_rmse, val_r2 = train_and_evaluate(df)

        # 3. Predict next week
        prediction = predict_future(model, df, feature_cols)

        recent_avg = df['ilitotal'].tail(8).mean()

        logger.info("\n" + "=" * 70)
        logger.info("NEXT WEEK PREDICTION")
        logger.info("=" * 70)
        logger.info("  Week Starting: %s",
                    prediction['week_start'].iloc[0].strftime('%Y-%m-%d'))
        logger.info("  Predicted ILI: %.0f", prediction['predicted_ili'].iloc[0])
        logger.info("  95%% PI: [%.0f, %.0f]",
                    prediction['ci_lower'].iloc[0],
                    prediction['ci_upper'].iloc[0])
        logger.info("  Recent 8-Week Avg: %.0f", recent_avg)
        logger.info("  Validation RMSE: %.2f", val_rmse)
        logger.info("=" * 70 + "\n")

        # 4. Check alert
        alert, threshold = check_alerts(prediction, recent_avg, THRESHOLD)

        # 5. Create visualization
        plot_buf = create_plot(df, prediction, threshold)

        # 6. Save artifacts
        save_artifacts(prediction, plot_buf)

        # 7. Send alert (with deduplication)
        if alert:
            logger.warning("⚠️  ALERT: Next week exceeds threshold!")
            logger.warning("  Week: %s", alert['week'].strftime('%Y-%m-%d'))
            logger.warning("  Predicted: %.0f cases", alert['predicted'])
            logger.warning("  Threshold: %.0f cases", alert['threshold'])
            logger.warning("  Excess: +%.0f cases (%.1f%%)",
                           alert['excess'],
                           alert['excess'] / alert['threshold'] * 100)

            if should_send_alert(alert):
                email_sent = send_email(alert, prediction, plot_buf, threshold)

                if email_sent:
                    with open(os.environ.get('GITHUB_OUTPUT', '/dev/null'), 'a') as f:
                        f.write("alert=true\n")
                        f.write(f"predicted={alert['predicted']:.0f}\n")
                        f.write(f"threshold={alert['threshold']:.0f}\n")
                        f.write(f"excess={alert['excess']:.0f}\n")
            else:
                logger.info("Alert suppressed by cooldown — email not sent")
        else:
            logger.info("✓ No alert: Prediction within normal range")
            logger.info("  Predicted: %.0f", prediction['predicted_ili'].iloc[0])
            logger.info("  Threshold: %.0f", threshold)
            logger.info("  Margin: %.0f cases below threshold",
                        threshold - prediction['predicted_ili'].iloc[0])

            with open(os.environ.get('GITHUB_OUTPUT', '/dev/null'), 'a') as f:
                f.write("alert=false\n")
                f.write(f"predicted={prediction['predicted_ili'].iloc[0]:.0f}\n")
                f.write(f"threshold={threshold:.0f}\n")

        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70 + "\n")

        return 0

    except Exception as e:
        logger.error("\n" + "=" * 70)
        logger.error("PIPELINE FAILED!")
        logger.error("=" * 70)
        logger.error("Error: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    exit(run_pipeline())
