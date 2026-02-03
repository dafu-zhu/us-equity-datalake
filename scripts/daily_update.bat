@echo off
REM ── Daily Update for US Equity Data Lake ──
REM Backfills last 3 days to cover missed runs (skips already-uploaded dates)
REM Scheduled via Windows Task Scheduler

cd /d D:\GitHub\us-equity-datalake

REM Compute dates via PowerShell (yesterday and 3 days ago)
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).AddDays(-3).ToString('yyyy-MM-dd')"') do set BACKFILL_FROM=%%i
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).AddDays(-1).ToString('yyyy-MM-dd')"') do set TARGET_DATE=%%i
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm'"') do set TIMESTAMP=%%i

echo ========================================>> logs\daily_update.log
echo [%TIMESTAMP%] Starting daily update (%BACKFILL_FROM% to %TARGET_DATE%)>> logs\daily_update.log
echo ========================================>> logs\daily_update.log

uv run quantdl-update --backfill-from %BACKFILL_FROM% --date %TARGET_DATE% --no-wrds >> logs\daily_update.log 2>&1

if %ERRORLEVEL% EQU 0 (
    echo [%TIMESTAMP%] Update completed successfully>> logs\daily_update.log
) else (
    echo [%TIMESTAMP%] Update failed with exit code %ERRORLEVEL%>> logs\daily_update.log
)
