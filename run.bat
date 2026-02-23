
@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] 仮想環境が見つかりません。先に setup.bat を実行してください。
  exit /b 1
)

echo [INFO] Swing Annotator GUI を起動します...
".venv\Scripts\python.exe" gui_app.py
exit /b %errorlevel%
