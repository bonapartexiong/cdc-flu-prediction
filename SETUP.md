# GitHub Secrets Setup — CDC Flu Prediction Pipeline

Follow this guide once after forking/cloning the repository. You only need
to do it the first time; secrets persist across all future workflow runs.

---

## Secrets you must create

Go to your GitHub repository → **Settings** → **Secrets and variables** →
**Actions** → **New repository secret**.  Add each of these three:

| Secret name | What to paste | Notes |
|---|---|---|
| `EMAIL_SENDER` | `yourname@gmail.com` | The Gmail account that **sends** the alert |
| `EMAIL_PASSWORD` | `abcd efgh ijkl mnop` | A 16‑character **Gmail App Password** (see below) |
| `EMAIL_RECIPIENTS` | `alice@example.com,bob@example.com` | Comma‑separated; no spaces after commas |

After adding all three, the **Actions → Secrets** page should look like this:

```
EMAIL_PASSWORD    ***
EMAIL_RECIPIENTS  alice@example.com,bob@example.com
EMAIL_SENDER      yourname@gmail.com
```

---

## How to get a Gmail App Password

A regular Gmail password will **not** work — Google blocks "less secure apps".
Instead you generate an app‑specific password:

1. Go to **https://myaccount.google.com/apppasswords**
2. Sign in with the Gmail account that will send alerts
3. If prompted, enable **2‑Step Verification** (required)
4. Click **Select app** → choose **Mail**
5. Click **Select device** → choose **Other** → type `CDC Flu Pipeline`
6. Click **Generate**
7. Copy the 16‑character password (format: `xxxx xxxx xxxx xxxx`)
8. Paste it into the `EMAIL_PASSWORD` secret **without** the spaces

> **Tip:** Create a dedicated Gmail account for this pipeline (e.g.
> `cdc.flu.alerts@gmail.com`). That way your personal account isn't tied
> to the automation, and you can revoke the app password independently.

---

## Optional secrets (defaults are fine for Gmail)

These are **not required** — only add them if you're using a non‑Gmail
SMTP provider or want to tune the alert threshold:

| Secret name | Default | When to change |
|---|---|---|
| `SMTP_SERVER` | `smtp.gmail.com` | Only if using Outlook / Yahoo / custom SMTP |
| `SMTP_PORT` | `587` | Only if your provider uses a different port |
| `THRESHOLD_MULTIPLIER` | `1.1` | Raise (1.2+) to get fewer alerts; lower (1.05) for earlier warnings |
| `STATE` | `California` | Change to any U.S. state name to track a different region |

---

## Verify it works

1. Go to the **Actions** tab in your repository
2. Select the **CDC Flu Prediction** workflow
3. Click **Run workflow** → **Run workflow** (manual trigger)
4. Wait ~2 minutes for the run to complete
5. Check the recipient inbox — you should receive a test email with a
   forecast chart embedded inline

If the email doesn't arrive:
- Check the **Spam** folder
- Confirm the App Password was pasted *without spaces*
- Look at the workflow run logs for "Email not configured" or SMTP errors

---

## What happens without email secrets?

The pipeline still runs normally — it fetches CDC data, trains the model,
generates a prediction, and uploads artifact files.  Email is skipped
gracefully with a log warning.  You'll still see results in:

- The **workflow run logs**
- The auto‑generated **Job Summary** on the run page
- A **GitHub Issue** (created automatically when the alert threshold is exceeded)
