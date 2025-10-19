# ============================================================================
# README.md - GitHub Actions Deployment Guide
# ============================================================================

# CDC Flu Prediction Pipeline - GitHub Actions

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
# - requirements.txt
# - .github/workflows/flu_prediction.yml
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
| `THRESHOLD_MULTIPLIER` | `1.1` | Alert threshold (1.1 = 10% above avg) |

### Step 4: Enable GitHub Actions

1. Go to **Actions** tab in your repository
2. Click "I understand my workflows, go ahead and enable them"
3. Your pipeline is now ready!