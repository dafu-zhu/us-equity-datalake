# GitHub Actions Secrets Setup

The `daily-update.yml` workflow requires the following secrets to be configured in your GitHub repository.

**Note**: As of January 2026, the workflow runs in **WRDS-free mode** to avoid IP restrictions on GitHub Actions runners. It uses Nasdaq FTP for universe data and SEC's public API for CIK mappings instead of WRDS/CRSP.

## Setting Up Secrets

1. Go to your repository on GitHub
2. Navigate to: **Settings** > **Secrets and variables** > **Actions**
3. Click **New repository secret** for each secret below

## Required Secrets

### Data Source Credentials (No WRDS Required)

| Secret Name | Description | Example |
|------------|-------------|---------|
| `ALPACA_API_KEY` | Alpaca Markets API key for ticks data | `PKXXXXXXXXXXXXX` |
| `ALPACA_API_SECRET` | Alpaca Markets API secret | `xxxxxxxxxxxxx` |
| `AWS_ACCESS_KEY_ID` | AWS access key for S3 storage | `AKIAXXXXXXXX` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key | `xxxxxxxxxxxxxxxx` |
| `SEC_USER_AGENT` | Your email for SEC EDGAR API | `your.name@example.com` |

### Email Notification Credentials

| Secret Name | Description | Example |
|------------|-------------|---------|
| `MAIL_SERVER` | SMTP server address | `smtp.gmail.com` |
| `MAIL_PORT` | SMTP port (usually 587 for TLS) | `587` |
| `MAIL_USERNAME` | Email address for sending | `your.email@gmail.com` |
| `MAIL_PASSWORD` | Email password or app password* | `abcd efgh ijkl mnop` |
| `MAIL_TO` | Recipient email address | `recipient@example.com` |
| `MAIL_FROM` | Sender email address | `your.email@gmail.com` |

**Important**: If using Gmail with 2-factor authentication enabled:
1. Go to https://myaccount.google.com/apppasswords
2. Select "Mail" and "Other (Custom name)"
3. Generate an app-specific password (16 characters)
4. Use this app password (without spaces) for `MAIL_PASSWORD`

## Verification

### Check Secrets Are Set
Run the test workflow to verify all secrets are configured:

1. Go to: **Actions** > **Test Daily Update**
2. Click **Run workflow**
3. Check the "Debug - Check secrets exist" step output

All secrets should show "SET" (not "MISSING").

### Test the Workflow
The test workflow runs `quantdl-update --no-ticks` which:
- Only checks for new fundamental filings (fast)
- Validates WRDS, AWS, and SEC credentials
- Sends a test email

If successful, the main `daily-update.yml` workflow will run daily at 9:00 UTC (4:00 AM ET).

## How It Works

### WRDS-Free Mode
The workflow runs with `--no-wrds` flag, which bypasses WRDS entirely:
- **Universe data**: Fetched from Nasdaq FTP (current stocks only)
- **CIK mapping**: Uses SEC's public API (`https://www.sec.gov/files/company_tickers.json`)
- **Ticks data**: Fetched from Alpaca API (unchanged)
- **Fundamentals**: Fetched from SEC EDGAR API (unchanged)

This mode is suitable for GitHub Actions where WRDS has IP-based restrictions that block dynamic runner IPs.

**Limitations of WRDS-free mode**:
- Uses current universe only (no historical universe from specific past dates)
- CIK mapping based on current SEC snapshot (may miss delisted/merged companies)
- Suitable for daily updates of currently-traded stocks

## Troubleshooting

### Error: "Application-specific password required"
- Using Gmail with 2FA but regular password instead of app password
- Generate app password at https://myaccount.google.com/apppasswords

### Error: "Invalid AWS credentials"
- `AWS_ACCESS_KEY_ID` or `AWS_SECRET_ACCESS_KEY` is incorrect
- Verify credentials in AWS IAM console
- Ensure IAM user has S3 read/write permissions

### Workflow doesn't run on schedule
- Scheduled workflows may be delayed up to 1 hour during high GitHub load
- GitHub disables scheduled workflows in repos with no activity for 60 days
- Check Actions tab for any failed runs or error messages
