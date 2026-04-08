@echo off
TITLE Scout Director Launcher
echo ===================================================
echo   Starting Scout Director... Please wait.
echo ===================================================

SET "MARKER=.python_env_ready"
SET "NEEDS_INSTALL=0"

if not exist %MARKER% (
    set "NEEDS_INSTALL=1"
) else (
    :: This clever trick uses xcopy to check if requirements.txt is newer than the marker
    xcopy /d /y "requirements.txt" "%MARKER%*" | findstr /i "1 File(s) copied" >nul
    if %errorlevel% equ 0 set "NEEDS_INSTALL=1"
)

if "%NEEDS_INSTALL%"=="1" (
    echo [INFO] Syncing Brewery dependencies...
    :: --quiet hides the "Requirement satisfied" spam
    :: --disable-pip-version-check prevents the annoying "New version available" text
    python_env\python.exe -m pip install -r requirements.txt --quiet --disable-pip-version-check --no-python-version-warning
    
    :: Create or update the marker so we skip this next time
    echo Ready > %MARKER%
    echo [INFO] Dependencies updated successfully.
) else (
    echo [INFO] Environment is up to date. Skipping sync.
)
:: 1. Set a default just in case Python fails
SET OLLAMA_URL=http://localhost:11434

:: Ask Portable Python for the URL from config.py
FOR /F "tokens=*" %%g IN ('python_env\python.exe -c "import config; print(config.OLLAMA_URL)" 2^>nul') do (SET OLLAMA_URL=%%g)

echo [INFO] Target AI Engine: %OLLAMA_URL%

:: 2. Route the logic using GOTO to avoid nested IF crashes
echo %OLLAMA_URL% | findstr /I "localhost 127.0.0.1" >nul
if %errorlevel% neq 0 goto REMOTE_CHECK

:LOCAL_CHECK
:: Check if local Ollama is responding
curl -s %OLLAMA_URL%/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Local AI Engine is already running.
    goto LAUNCH_APP
)


echo [INFO] Starting local AI Engine (Ollama)...
start "" /min ollama serve
timeout /t 3 /nobreak >nul


goto LAUNCH_APP

:REMOTE_CHECK
:: Just ping the remote server
curl -s %OLLAMA_URL%/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Cannot reach remote AI Engine at %OLLAMA_URL%
    echo Please ensure your remote server is active.
)


:LAUNCH_APP
:: Embedding models are pulled on-demand by the app when the user selects one.

:: 3. Run the PyWebView wrapper
echo [INFO] Launching Interface...
python_env\python.exe scout_desktop.py

:: 4. The Crash Trap
echo.
echo [DEBUG] The application has closed. If it didn't open at all, read the error above!
pause