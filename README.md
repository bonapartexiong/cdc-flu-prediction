# ============================================================================
# README.md - GitHub Actions Deployment Guide
# ============================================================================

# CDC Flu Prediction Pipeline - GitHub Actions

> Automated weekly flu forecasting with email reports: an 8-week case chart
> plus a red-line alert when the latest week exceeds the predicted volume by
> more than 3%.

## 🚀 Quick Setup (5 Minutes)

### Step 1: Get Gmail App Password

1. Go to: https://myaccount.google.com/apppasswords
2. Enable 2-Factor Authentication (if not already enabled)
3. Click "Select app" → Choose "Mail"
4. Click "Select device" → Choose "Other" → Enter "CDC Flu Pipeline"
5. Click "Generate"
6. **Copy the 16-character password** (e.g., `abcd efgh ijkl mnop`)

### Step 2: Fork or Clone Repository

```bash
# Create new repository
mkdir cdc-flu-prediction
cd cdc-flu-prediction

# Initialize git
git init

# Copy these files:
# - cdc_flu_pipeline.py
# - test.py
# - requirements.txt
# - .github/workflows/flu_prediction.yml
# - state/.gitkeep
# - README.md

# Commit
git add .
git commit -m "Initial commit: CDC Flu Prediction Pipeline"

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/cdc-flu-prediction.git
git branch -M main
git push -u origin main
```

### Step 3: Configure GitHub Secrets

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add these secrets one by one:

| Secret Name | Value | Example |
|-------------|-------|---------|
| `EMAIL_SENDER` | Your Gmail address | `yourname@gmail.com` |
| `EMAIL_PASSWORD` | Gmail app password | `abcd efgh ijkl mnop` |
| `EMAIL_RECIPIENTS` | Comma-separated emails | `user1@example.com,user2@example.com` |

**Optional secrets** (uses defaults if not set):
| Secret Name | Default | Description |
|-------------|---------|-------------|
| `SMTP_SERVER` | `smtp.gmail.com` | SMTP server |
| `SMTP_PORT` | `587` | SMTP port |
| `STATE` | `California` | US state to monitor |
| `EXCEED_THRESHOLD` | `0.03` | Red-line alert: latest week this far above predicted volume (0.03 = 3%) |

### Step 4: Enable GitHub Actions

1. Go to **Actions** tab in your repository
2. Click "I understand my workflows, go ahead and enable them"
3. Your pipeline is now ready!

## 📧 Weekly Email Report

Every Monday the pipeline fetches the latest CDC flu data, retrains the
model, and emails a chart of **weekly flu cases over the past 8 weeks**.

If the **last data point** (the newest week of actual cases) exceeds the
volume predicted for that same week by more than `EXCEED_THRESHOLD`
(3 % by default), a **vertical red line** is drawn at that point and the
email subject is flagged `⚠️ ABOVE PREDICTION`.

> 💡 The first run is a "cold start" (no prior prediction exists yet), so the
> red-line alert begins on the second weekly run. After each run the pipeline
> commits `state/last_prediction.json` back to the repository so the next
> run can compare the latest actual week against the prior forecast.

### Testing locally

```bash
pip install -r requirements.txt
pytest test.py -v
```
