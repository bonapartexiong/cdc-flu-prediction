"""
CMU Delphi FluView Prediction Pipeline - Optimized for GitHub Actions
Automated flu forecasting with email alerts using CMU Delphi Epidata API
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

# Configuration from environment variables
DELPHI_API = "https://api.delphi.cmu.edu/epidata/fluview"
REGION = "ca"  # California state code (lowercase for Delphi API)
THRESHOLD = float(os.getenv("THRESHOLD_MULTIPLIER", "1.15"))

# Email config
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_FROM = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_RECIPIENTS", "").split(",")


def get_epiweek_range(years_back=3):
    """
    Calculate epiweek range for API query
    Epiweeks are in YYYYWW format (e.g., 202401 = 2024 week 1)
    """
    current_date = datetime.now()
    current_year = current_date.year
    current_week = current_date.isocalendar()[1]
    
    # Current epiweek
    current_epiweek = current_year * 100 + current_week
    
    # Start epiweek (years_back years ago)
    start_year = current_year - years_back
    start_epiweek = start_year * 100 + 1  # Week 1 of that year
    
    return start_epiweek, current_epiweek


def fetch_flu_data():
    """Fetch flu data from CMU Delphi Epidata API"""
    logger.info(f"Fetching flu data for region: {REGION}...")
    
    try:
        # Get epiweek range (last 3 years)
        start_epiweek, end_epiweek = get_epiweek_range(years_back=3)
        
        # Delphi API parameters
        params = {
            "regions": REGION,
            "epiweeks": f"{start_epiweek}-{end_epiweek}"
        }
        
        logger.info(f"Requesting data for epiweeks {start_epiweek} to {end_epiweek}")
        logger.info(f"API URL: {DELPHI_API}")
        logger.info(f"Parameters: {params}")
        
        response = requests.get(DELPHI_API, params=params, timeout=30)
        
        logger.info(f"Status Code: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"API Response: {response.text[:500]}")
            response.raise_for_status()
        
        data = response.json()
        
        # Check API response structure
        if 'result' not in data:
            raise ValueError(f"Unexpected API response format: {data}")
        
        if data['result'] != 1:
            error_msg = data.get('message', 'Unknown error')
            raise ValueError(f"API returned error: {error_msg}")
        
        if 'epidata' not in data or not data['epidata']:
            raise ValueError(f"No data returned for region: {REGION}")
        
        epidata = data['epidata']
        logger.info(f"Received {len(epidata)} records")
        
        # Create DataFrame
        df = pd.DataFrame(epidata)
        
        # Show available columns
        logger.info(f"Available columns: {df.columns.tolist()}")
        
        # Convert epiweek to date
        df['epiweek'] = df['epiweek'].astype(int)
        df['year'] = df['epiweek'] // 100
        df['week'] = df['epiweek'] % 100
        
        # Create date from epiweek (approximate - first day of week)
        df['date'] = pd.to_datetime(
            df['year'].astype(str) + '-W' + df['week'].astype(str).str.zfill(2) + '-1',
            format='%Y-W%W-%w'
        )
        
        # Convert numeric columns
        # Delphi API provides: wili (weighted ILI %), ili, num_ili, num_patients, num_providers
        numeric_cols = ['wili', 'ili', 'num_ili', 'num_patients', 'num_providers']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Use num_ili (number of ILI cases) as target if available, otherwise calculate
        if 'num_ili' in df.columns:
            df['ili_cases'] = df['num_ili']
        elif 'ili' in df.columns and 'num_patients' in df.columns:
            # Calculate from percentage
            df['ili_cases'] = (df['ili'] / 100) * df['num_patients']
        else:
            raise ValueError("Cannot determine ILI case count from available columns")
        
        # Remove rows with null critical values
        df = df.dropna(subset=['date', 'ili_cases'])
        
        # Remove zero or negative cases
        df = df[df['ili_cases'] > 0]
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Remove duplicates (keep latest)
        df = df.drop_duplicates(subset=['epiweek'], keep='last')
        df = df.sort_values('date').reset_index(drop=True)
        
        if len(df) < 52:
            logger.warning(f"Only {len(df)} weeks of data available (need 52+ for good predictions)")
        
        logger.info(f"✓ Fetched {len(df)} weeks of clean data")
        logger.info(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        logger.info(f"  ILI case range: {df['ili_cases'].min():.0f} to {df['ili_cases'].max():.0f}")
        
        # Show sample of recent data
        logger.info(f"\nRecent data sample:")
        recent = df.tail(3)[['date', 'epiweek', 'ili_cases', 'ili', 'num_patients']]
        for _, row in recent.iterrows():
            logger.info(f"  {row['date'].date()} (epiweek {row['epiweek']}): {row['ili_cases']:.0f} cases")
        
        return df
        
    except requests.exceptions.Timeout:
        logger.error("Request timed out - Delphi API may be slow")
        raise
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error fetching flu data: {e}", exc_info=True)
        raise


def create_features(df):
    """Create time series features for flu prediction"""
    df = df.copy().sort_values('date').reset_index(drop=True)
    target = 'ili_cases'
    
    if len(df) < 12:
        logger.warning(f"Only {len(df)} weeks of data - some features may be incomplete")
    
    # Lagged features (1-8 weeks back)
    for lag in range(1, 9):
        df[f'lag_{lag}'] = df[target].shift(lag)
    
    # Rolling statistics
    for window in [2, 4, 8, 12]:
        df[f'roll_mean_{window}'] = df[target].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'roll_std_{window}'] = df[target].shift(1).rolling(window=window, min_periods=1).std()
        df[f'roll_max_{window}'] = df[target].shift(1).rolling(window=window, min_periods=1).max()
        df[f'roll_min_{window}'] = df[target].shift(1).rolling(window=window, min_periods=1).min()
    
    # Seasonal features (flu has strong seasonality)
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    
    # Cyclical encoding for week
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # Year-over-year comparison (52 weeks ago)
    df['lag_52'] = df[target].shift(52)
    
    # Trend features
    df['trend_2w'] = df[target].shift(1) - df[target].shift(3)
    df['trend_4w'] = df[target].shift(1) - df[target].shift(5)
    
    # Growth rate
    df['growth_rate'] = (df[target].shift(1) - df[target].shift(2)) / (df[target].shift(2) + 1)
    
    # Days since start
    df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
    
    # Is flu season (October-March in Northern Hemisphere)
    df['is_flu_season'] = df['month'].isin([10, 11, 12, 1, 2, 3]).astype(int)
    
    # Drop rows with NaN in critical features
    df_clean = df.dropna()
    
    logger.info(f"  Features created: {len(df)} → {len(df_clean)} usable rows after dropna")
    
    return df_clean, target


def train_model(df):
    """Train LightGBM model for flu prediction"""
    logger.info("Training prediction model...")
    
    df_feat, target = create_features(df)
    
    if len(df_feat) < 20:
        raise ValueError(f"Insufficient training data: only {len(df_feat)} rows after feature engineering")
    
    # Define feature columns
    exclude_cols = ['date', 'epiweek', 'year', 'week', 'ili_cases', 
                    'ili', 'wili', 'num_ili', 'num_patients', 'num_providers',
                    'region', 'lag_year']
    feature_cols = [c for c in df_feat.columns if c not in exclude_cols]
    
    X = df_feat[feature_cols]
    y = df_feat[target]
    
    logger.info(f"  Training with {len(X)} samples and {len(feature_cols)} features")
    
    # Train model with parameters tuned for flu seasonality
    model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=64,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X, y)
    
    # Evaluate
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mae = np.mean(np.abs(y - y_pred))
    mape = np.mean(np.abs((y - y_pred) / (y + 1))) * 100
    
    logger.info(f"✓ Model trained successfully")
    logger.info(f"  RMSE: {rmse:.2f}")
    logger.info(f"  MAE: {mae:.2f}")
    logger.info(f"  MAPE: {mape:.2f}%")
    logger.info(f"  R²: {r2:.3f}")
    
    # Feature importance (top 5)
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"  Top features: {', '.join(importance.head(5)['feature'].tolist())}")
    
    return model, feature_cols


def predict_future(model, df, feature_cols):
    """Generate next week prediction"""
    logger.info("Generating next week prediction...")
    
    df_feat, target = create_features(df)
    
    if len(df_feat) == 0:
        raise ValueError("No data available for prediction after feature engineering")
    
    # Get features for last available week
    X_pred = df_feat[feature_cols].iloc[-1:].copy()
    
    # Check for missing features
    if X_pred.isnull().any().any():
        logger.warning("Missing values in prediction features - filling with column means")
        for col in X_pred.columns:
            if X_pred[col].isnull().any():
                X_pred[col] = X_pred[col].fillna(df_feat[col].mean())
    
    # Make prediction
    pred_value = model.predict(X_pred)[0]
    pred_value = max(0, pred_value)  # Can't have negative cases
    
    # Calculate confidence interval based on recent prediction errors
    recent_window = min(30, len(df_feat))
    recent = df_feat.tail(recent_window)
    X_recent = recent[feature_cols]
    recent_preds = model.predict(X_recent)
    residuals = recent[target].values - recent_preds
    std = np.std(residuals)
    
    # Next week date
    last_date = df['date'].max()
    next_date = last_date + timedelta(weeks=1)
    
    # Calculate next epiweek
    last_epiweek = df['epiweek'].max()
    last_year = last_epiweek // 100
    last_week = last_epiweek % 100
    
    if last_week == 52:
        next_epiweek = (last_year + 1) * 100 + 1
    else:
        next_epiweek = last_epiweek + 1
    
    prediction = {
        'date': next_date,
        'epiweek': next_epiweek,
        'predicted_ili_cases': pred_value,
        'ci_lower': max(0, pred_value - 1.96 * std),
        'ci_upper': pred_value + 1.96 * std,
        'std_error': std
    }
    
    logger.info(f"✓ Prediction generated for epiweek {next_epiweek} (week starting {next_date.date()})")
    
    return pd.DataFrame([prediction])


def check_alerts(prediction, recent_avg, threshold_mult=1.15):
    """Check if prediction exceeds threshold"""
    threshold = recent_avg * threshold_mult
    
    alert = None
    predicted = prediction['predicted_ili_cases'].iloc[0]
    
    if predicted > threshold:
        alert = {
            'week': prediction['date'].iloc[0],
            'epiweek': prediction['epiweek'].iloc[0],
            'predicted': predicted,
            'threshold': threshold,
            'excess': predicted - threshold,
            'recent_avg': recent_avg,
            'percent_above': (predicted - threshold) / threshold * 100
        }
    
    return alert, threshold


def create_plot(historical, prediction, threshold):
    """Create forecast visualization"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Historical data (last year or all data)
    hist_recent = historical.tail(52)
    ax.plot(hist_recent['date'], hist_recent['ili_cases'],
            label='Historical ILI Cases', marker='o', linewidth=2, color='#2196F3')
    
    # Single week prediction
    ax.plot(prediction['date'], prediction['predicted_ili_cases'],
            label='Next Week Prediction', marker='s', markersize=12, linewidth=2.5,
            linestyle='--', color='#F44336')
    
    # Confidence interval as error bar
    ax.errorbar(prediction['date'].iloc[0], 
                prediction['predicted_ili_cases'].iloc[0],
                yerr=[[prediction['predicted_ili_cases'].iloc[0] - prediction['ci_lower'].iloc[0]], 
                      [prediction['ci_upper'].iloc[0] - prediction['predicted_ili_cases'].iloc[0]]],
                fmt='none', color='#F44336', capsize=10, capthick=2,
                label='95% Confidence Interval', alpha=0.6)
    
    # Threshold line
    ax.axhline(y=threshold, color='#FF9800', linestyle=':',
               linewidth=2.5, label=f'Alert Threshold ({threshold:.0f})')
    
    # Highlight if exceeds threshold
    if prediction['predicted_ili_cases'].iloc[0] > threshold:
        ax.scatter(prediction['date'], prediction['predicted_ili_cases'],
                  color='#F44336', s=300, zorder=5, marker='*',
                  edgecolors='darkred', linewidths=2,
                  label='⚠️ Threshold Exceeded')
    
    ax.set_xlabel('Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('ILI Cases', fontsize=12, fontweight='bold')
    ax.set_title(f'California Flu Activity - Next Week Forecast\nGenerated: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


def send_email(alert, prediction, plot_buf, threshold):
    """Send email alert"""
    if not EMAIL_FROM or not EMAIL_PASSWORD or not EMAIL_TO[0]:
        logger.warning("Email not configured - skipping notification")
        return False
    
    logger.info(f"Sending email to {len(EMAIL_TO)} recipient(s)...")
    
    # Create email
    msg = MIMEMultipart('related')
    msg['Subject'] = f'🚨 Flu Alert: Next week exceeds threshold - California'
    msg['From'] = EMAIL_FROM
    msg['To'] = ', '.join(EMAIL_TO)
    
    # HTML body
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
            .alert {{ background: #fff3cd; padding: 20px; border-left: 5px solid #ffc107; margin: 20px 0; }}
            .metric {{ font-size: 28px; font-weight: bold; color: #d32f2f; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background: #f2f2f2; font-weight: bold; }}
            .exceed {{ color: #d32f2f; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h2>🚨 Flu Activity Alert - California</h2>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        
        <div class="alert">
            <h3>⚠️ Threshold Exceeded for Next Week</h3>
            <p><span class="metric">{alert['predicted']:.0f}</span> predicted ILI cases</p>
            <p>This is <strong>{alert['excess']:.0f} cases</strong> above the alert threshold 
            ({alert['percent_above']:.1f}% above threshold)</p>
        </div>
        
        <h3>📊 Next Week Forecast:</h3>
        <table>
            <tr>
                <th>Epiweek</th>
                <th>Week Starting</th>
                <th>Predicted ILI Cases</th>
                <th>95% Confidence Interval</th>
                <th>Alert Threshold</th>
                <th>Recent 8-Week Avg</th>
            </tr>
            <tr>
                <td><strong>{alert['epiweek']}</strong></td>
                <td><strong>{prediction['date'].iloc[0].strftime('%Y-%m-%d')}</strong></td>
                <td class="exceed">{prediction['predicted_ili_cases'].iloc[0]:.0f}</td>
                <td>{prediction['ci_lower'].iloc[0]:.0f} - {prediction['ci_upper'].iloc[0]:.0f}</td>
                <td>{alert['threshold']:.0f}</td>
                <td>{alert['recent_avg']:.0f}</td>
            </tr>
        </table>
        
        <h3>📈 Key Metrics:</h3>
        <ul>
            <li><strong>Predicted Cases:</strong> {alert['predicted']:.0f}</li>
            <li><strong>Alert Threshold:</strong> {alert['threshold']:.0f} ({THRESHOLD*100:.0f}% of recent average)</li>
            <li><strong>Excess Cases:</strong> +{alert['excess']:.0f} ({alert['percent_above']:.1f}% above threshold)</li>
            <li><strong>Confidence Range:</strong> {prediction['ci_lower'].iloc[0]:.0f} to {prediction['ci_upper'].iloc[0]:.0f}</li>
            <li><strong>Recent 8-Week Average:</strong> {alert['recent_avg']:.0f}</li>
        </ul>
        
        <p style="margin-top: 30px; color: #666; font-size: 12px;">
            <em>This is an automated alert from the CMU Delphi Flu Prediction System (GitHub Actions)</em><br>
            Data source: CMU Delphi Epidata API (FluView) | Region: California<br>
            See attached chart for visualization
        </p>
    </body>
    </html>
    """
    
    msg.attach(MIMEText(html, 'html'))
    
    # Attach plot
    img = MIMEImage(plot_buf.getvalue())
    img.add_header('Content-ID', '<forecast_plot>')
    img.add_header('Content-Disposition', 'attachment', filename='flu_forecast.png')
    msg.attach(img)
    
    # Send
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info("✓ Email sent successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


def save_artifacts(prediction, plot_buf):
    """Save artifacts for GitHub Actions"""
    try:
        os.makedirs('output', exist_ok=True)
        
        # Save prediction as CSV
        prediction.to_csv('output/prediction.csv', index=False)
        logger.info("✓ Prediction saved to output/prediction.csv")
        
        # Save plot
        with open('output/forecast_plot.png', 'wb') as f:
            f.write(plot_buf.getvalue())
        logger.info("✓ Plot saved to output/forecast_plot.png")
        
        # Save summary
        with open('output/summary.txt', 'w') as f:
            f.write(f"CMU Delphi Flu Prediction Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            f.write(f"Region: California\n\n")
            f.write(f"Next Week Prediction:\n")
            f.write(f"  Epiweek: {prediction['epiweek'].iloc[0]}\n")
            f.write(f"  Week Starting: {prediction['date'].iloc[0].strftime('%Y-%m-%d')}\n")
            f.write(f"  Predicted ILI Cases: {prediction['predicted_ili_cases'].iloc[0]:.0f}\n")
            f.write(f"  95% CI: [{prediction['ci_lower'].iloc[0]:.0f}, {prediction['ci_upper'].iloc[0]:.0f}]\n")
        logger.info("✓ Summary saved to output/summary.txt")
        
        return True
    except Exception as e:
        logger.error(f"Failed to save artifacts: {e}")
        return False


def run_pipeline():
    """Main pipeline execution"""
    logger.info("=" * 70)
    logger.info("CMU DELPHI FLU PREDICTION PIPELINE - GITHUB ACTIONS")
    logger.info("=" * 70)
    
    try:
        # 1. Fetch data
        df = fetch_flu_data()
        
        # 2. Train model
        model, feature_cols = train_model(df)
        
        # 3. Generate next week prediction
        prediction = predict_future(model, df, feature_cols)
        
        # Calculate recent average for threshold
        recent_avg = df['ili_cases'].tail(8).mean()
        
        logger.info(f"\n{'='*70}")
        logger.info("NEXT WEEK PREDICTION")
        logger.info(f"{'='*70}")
        logger.info(f"  Epiweek: {prediction['epiweek'].iloc[0]}")
        logger.info(f"  Week Starting: {prediction['date'].iloc[0].strftime('%Y-%m-%d')}")
        logger.info(f"  Predicted ILI Cases: {prediction['predicted_ili_cases'].iloc[0]:.0f}")
        logger.info(f"  95% CI: [{prediction['ci_lower'].iloc[0]:.0f}, {prediction['ci_upper'].iloc[0]:.0f}]")
        logger.info(f"  Recent 8-Week Avg: {recent_avg:.0f}")
        logger.info(f"{'='*70}\n")
        
        # 4. Check alert
        alert, threshold = check_alerts(prediction, recent_avg, THRESHOLD)
        
        # 5. Create visualization
        plot_buf = create_plot(df, prediction, threshold)
        
        # 6. Save artifacts
        save_artifacts(prediction, plot_buf)
        
        # 7. Send alert if needed
        if alert:
            logger.warning(f"⚠️  ALERT: Next week exceeds threshold!")
            logger.warning(f"  Epiweek: {alert['epiweek']}")
            logger.warning(f"  Week: {alert['week'].strftime('%Y-%m-%d')}")
            logger.warning(f"  Predicted: {alert['predicted']:.0f} cases")
            logger.warning(f"  Threshold: {alert['threshold']:.0f} cases")
            logger.warning(f"  Excess: +{alert['excess']:.0f} cases ({alert['percent_above']:.1f}%)")
            
            email_sent = send_email(alert, prediction, plot_buf, threshold)
            
            with open(os.environ.get('GITHUB_OUTPUT', '/dev/null'), 'a') as f:
                f.write(f"alert=true\n")
                f.write(f"predicted={alert['predicted']:.0f}\n")
                f.write(f"threshold={alert['threshold']:.0f}\n")
                f.write(f"excess={alert['excess']:.0f}\n")
        else:
            logger.info("✓ No alert: Prediction within normal range")
            logger.info(f"  Predicted: {prediction['predicted_ili_cases'].iloc[0]:.0f}")
            logger.info(f"  Threshold: {threshold:.0f}")
            logger.info(f"  Margin: {(threshold - prediction['predicted_ili_cases'].iloc[0]):.0f} cases below threshold")
            
            with open(os.environ.get('GITHUB_OUTPUT', '/dev/null'), 'a') as f:
                f.write(f"alert=false\n")
                f.write(f"predicted={prediction['predicted_ili_cases'].iloc[0]:.0f}\n")
                f.write(f"threshold={threshold:.0f}\n")
        
        logger.info(f"\n{'='*70}")
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"{'='*70}\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n{'='*70}")
        logger.error("PIPELINE FAILED!")
        logger.error(f"{'='*70}")
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(run_pipeline())
