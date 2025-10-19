"""
CDC FluView Prediction Pipeline - Optimized for GitHub Actions
Automated flu forecasting with email alerts for California
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
CDC_API = "https://data.cdc.gov/resource/pwn4-m3yp.json"
STATE = "California"
THRESHOLD = float(os.getenv("THRESHOLD_MULTIPLIER", "1.1"))

# Email config
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_FROM = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_RECIPIENTS", "").split(",")


def fetch_cdc_data():
    """Fetch flu data from CDC API"""
    logger.info(f"Fetching CDC data for {STATE}...")
    
    params = {
        "$limit": 5000,
        "$order": "week_start DESC",
        "$where": f"state_name='{STATE}'",
        "$select": "week_start,week,ilitotal,total_patients,percent_ili"
    }
    
    try:
        response = requests.get(CDC_API, params=params, timeout=30)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        
        df['week_start'] = pd.to_datetime(df['week_start'])
        df['week'] = df['week'].astype(int)
        df['ilitotal'] = pd.to_numeric(df['ilitotal'], errors='coerce')
        df['percent_ili'] = pd.to_numeric(df['percent_ili'], errors='coerce')
        
        df = df.sort_values('week_start').reset_index(drop=True)
        logger.info(f"✓ Fetched {len(df)} weeks of data ({df['week_start'].min().date()} to {df['week_start'].max().date()})")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch CDC data: {e}")
        raise


def create_features(df):
    """Create time series features"""
    df = df.copy().sort_values('week_start').reset_index(drop=True)
    target = 'ilitotal'
    
    # Lagged features
    for lag in range(1, 9):
        df[f'lag_{lag}'] = df[target].shift(lag)
    
    # Rolling statistics
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


def train_model(df):
    """Train LightGBM model"""
    logger.info("Training model...")
    
    df_feat, target = create_features(df)
    
    feature_cols = [c for c in df_feat.columns 
                   if c not in ['week_start', 'week', 'ilitotal', 'total_patients', 'percent_ili']]
    
    X = df_feat[feature_cols]
    y = df_feat[target]
    
    # Train model
    model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=50,
        min_child_samples=30,
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
    
    logger.info(f"✓ Model trained - RMSE: {rmse:.2f}, R²: {r2:.3f}")
    
    return model, feature_cols


def predict_future(model, df, feature_cols):
    """Generate next week prediction"""
    logger.info("Predicting next week...")
    
    df_feat, _ = create_features(df)
    target = 'ilitotal'
    
    if len(df_feat) == 0:
        logger.error("No data available for prediction")
        return pd.DataFrame()
    
    X_pred = df_feat[feature_cols].iloc[-1:].values
    pred_value = max(0, model.predict(X_pred)[0])
    
    # Confidence interval
    recent = df_feat.tail(30)
    X_recent = recent[feature_cols]
    recent_preds = model.predict(X_recent)
    std = np.std(recent[target] - recent_preds)
    
    last_date = df['week_start'].max()
    next_date = last_date + timedelta(weeks=1)
    
    prediction = {
        'week_start': next_date,
        'predicted_ili': pred_value,
        'ci_lower': max(0, pred_value - 1.96 * std),
        'ci_upper': pred_value + 1.96 * std
    }
    
    return pd.DataFrame([prediction])


def check_alerts(prediction, recent_avg, threshold_mult=1.1):
    """Check if prediction exceeds threshold"""
    threshold = recent_avg * threshold_mult
    
    alert = None
    if prediction['predicted_ili'].iloc[0] > threshold:
        alert = {
            'week': prediction['week_start'].iloc[0],
            'predicted': prediction['predicted_ili'].iloc[0],
            'threshold': threshold,
            'excess': prediction['predicted_ili'].iloc[0] - threshold,
            'recent_avg': recent_avg
        }
    
    return alert, threshold


def create_plot(historical, prediction, threshold):
    """Create forecast visualization"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Historical data (last year)
    hist_recent = historical.tail(52)
    ax.plot(hist_recent['week_start'], hist_recent['ilitotal'],
            label='Historical ILI Cases', marker='o', linewidth=2, color='#2196F3')
    
    # Single week prediction
    ax.plot(prediction['week_start'], prediction['predicted_ili'],
            label='Next Week Prediction', marker='s', markersize=12, linewidth=2.5,
            linestyle='--', color='#F44336')
    
    # Confidence interval as error bar
    ax.errorbar(prediction['week_start'].iloc[0], 
                prediction['predicted_ili'].iloc[0],
                yerr=[[prediction['predicted_ili'].iloc[0] - prediction['ci_lower'].iloc[0]], 
                      [prediction['ci_upper'].iloc[0] - prediction['predicted_ili'].iloc[0]]],
                fmt='none', color='#F44336', capsize=10, capthick=2,
                label='95% Confidence Interval', alpha=0.6)
    
    # Threshold line
    ax.axhline(y=threshold, color='#FF9800', linestyle=':',
               linewidth=2.5, label=f'Alert Threshold ({threshold:.0f})')
    
    # Highlight if exceeds threshold
    if prediction['predicted_ili'].iloc[0] > threshold:
        ax.scatter(prediction['week_start'], prediction['predicted_ili'],
                  color='#F44336', s=300, zorder=5, marker='*',
                  edgecolors='darkred', linewidths=2,
                  label='⚠️ Threshold Exceeded')
    
    ax.set_xlabel('Week Starting', fontsize=12, fontweight='bold')
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
    msg['Subject'] = f'🚨 Flu Alert: Next week exceeds threshold - {STATE}'
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
        <h2>🚨 Flu Activity Alert - {STATE}</h2>
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
                <th>95% Confidence Interval</th>
                <th>Alert Threshold</th>
                <th>Recent 8-Week Avg</th>
            </tr>
            <tr>
                <td><strong>{prediction['week_start'].iloc[0].strftime('%Y-%m-%d')}</strong></td>
                <td class="exceed">{prediction['predicted_ili'].iloc[0]:.0f}</td>
                <td>{prediction['ci_lower'].iloc[0]:.0f} - {prediction['ci_upper'].iloc[0]:.0f}</td>
                <td>{alert['threshold']:.0f}</td>
                <td>{alert['recent_avg']:.0f}</td>
            </tr>
        </table>
        
        <h3>📈 Key Metrics:</h3>
        <ul>
            <li><strong>Predicted Cases:</strong> {alert['predicted']:.0f}</li>
            <li><strong>Alert Threshold:</strong> {alert['threshold']:.0f} ({THRESHOLD*100:.0f}% of recent average)</li>
            <li><strong>Excess Cases:</strong> +{alert['excess']:.0f} ({(alert['excess']/alert['threshold']*100):.1f}%)</li>
            <li><strong>Confidence Range:</strong> {prediction['ci_lower'].iloc[0]:.0f} to {prediction['ci_upper'].iloc[0]:.0f}</li>
            <li><strong>Recent 8-Week Average:</strong> {alert['recent_avg']:.0f}</li>
        </ul>
        
        <p style="margin-top: 30px; color: #666; font-size: 12px;">
            <em>This is an automated alert from the CDC Flu Prediction System (GitHub Actions)</em><br>
            Data source: CDC FluView API | State: California<br>
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
        # Create output directory
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
            f.write(f"CDC Flu Prediction Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            f.write(f"State: {STATE}\n\n")
            f.write(f"Next Week Prediction:\n")
            f.write(f"  Week Starting: {prediction['week_start'].iloc[0].strftime('%Y-%m-%d')}\n")
            f.write(f"  Predicted ILI: {prediction['predicted_ili'].iloc[0]:.0f}\n")
            f.write(f"  95% CI: [{prediction['ci_lower'].iloc[0]:.0f}, {prediction['ci_upper'].iloc[0]:.0f}]\n")
        logger.info("✓ Summary saved to output/summary.txt")
        
        return True
    except Exception as e:
        logger.error(f"Failed to save artifacts: {e}")
        return False


def run_pipeline():
    """Main pipeline execution"""
    logger.info("=" * 70)
    logger.info("CDC FLU PREDICTION PIPELINE - GITHUB ACTIONS")
    logger.info("=" * 70)
    
    try:
        # 1. Fetch data
        df = fetch_cdc_data()
        
        # 2. Train model
        model, feature_cols = train_model(df)
        
        # 3. Generate next week prediction
        prediction = predict_future(model, df, feature_cols)
        
        # Calculate recent average for threshold
        recent_avg = df['ilitotal'].tail(8).mean()
        
        logger.info(f"\n{'='*70}")
        logger.info("NEXT WEEK PREDICTION")
        logger.info(f"{'='*70}")
        logger.info(f"  Week Starting: {prediction['week_start'].iloc[0].strftime('%Y-%m-%d')}")
        logger.info(f"  Predicted ILI: {prediction['predicted_ili'].iloc[0]:.0f}")
        logger.info(f"  95% CI: [{prediction['ci_lower'].iloc[0]:.0f}, {prediction['ci_upper'].iloc[0]:.0f}]")
        logger.info(f"  Recent 8-Week Avg: {recent_avg:.0f}")
        logger.info(f"{'='*70}\n")
        
        # 4. Check alert
        alert, threshold = check_alerts(prediction, recent_avg, THRESHOLD)
        
        # 5. Create visualization
        plot_buf = create_plot(df, prediction, threshold)
        
        # 6. Save artifacts for GitHub Actions
        save_artifacts(prediction, plot_buf)
        
        # 7. Send alert if needed
        if alert:
            logger.warning(f"⚠️  ALERT: Next week exceeds threshold!")
            logger.warning(f"  Week: {alert['week'].strftime('%Y-%m-%d')}")
            logger.warning(f"  Predicted: {alert['predicted']:.0f} cases")
            logger.warning(f"  Threshold: {alert['threshold']:.0f} cases")
            logger.warning(f"  Excess: +{alert['excess']:.0f} cases ({(alert['excess']/alert['threshold']*100):.1f}%)")
            
            email_sent = send_email(alert, prediction, plot_buf, threshold)
            
            if email_sent:
                # Set GitHub Actions output
                with open(os.environ.get('GITHUB_OUTPUT', '/dev/null'), 'a') as f:
                    f.write(f"alert=true\n")
                    f.write(f"predicted={alert['predicted']:.0f}\n")
                    f.write(f"threshold={alert['threshold']:.0f}\n")
                    f.write(f"excess={alert['excess']:.0f}\n")
        else:
            logger.info("✓ No alert: Prediction within normal range")
            logger.info(f"  Predicted: {prediction['predicted_ili'].iloc[0]:.0f}")
            logger.info(f"  Threshold: {threshold:.0f}")
            logger.info(f"  Margin: {(threshold - prediction['predicted_ili'].iloc[0]):.0f} cases below threshold")
            
            # Set GitHub Actions output
            with open(os.environ.get('GITHUB_OUTPUT', '/dev/null'), 'a') as f:
                f.write(f"alert=false\n")
                f.write(f"predicted={prediction['predicted_ili'].iloc[0]:.0f}\n")
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