@echo off
setlocal
cd /d "%~dp0"
set "MABI_PY="
if exist ".venv\Scripts\python.exe" set "MABI_PY=.venv\Scripts\python.exe"
if defined MABI_PY goto executar
if exist "%LocalAppData%\Programs\Python\Python314\python.exe" set "MABI_PY=%LocalAppData%\Programs\Python\Python314\python.exe"
if defined MABI_PY goto dependencias
py -3 -c "import sys; raise SystemExit(sys.version_info < (3, 11))" >nul 2>&1
if not errorlevel 1 (
  for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)"') do set "MABI_PY=%%P"
  goto dependencias
)
python -c "import sys; raise SystemExit(sys.version_info < (3, 11))" >nul 2>&1
if not errorlevel 1 set "MABI_PY=python"
if not defined MABI_PY (
  echo Instale o Python 3.11 ou mais recente para iniciar o painel.
  pause
  exit /b 1
)

:dependencias
"%MABI_PY%" -c "import pandas, numpy, pyarrow, scipy, sklearn, xgboost, optuna, joblib, starlette, uvicorn, multipart" >nul 2>&1
if not errorlevel 1 goto executar
"%MABI_PY%" -m venv .venv
if errorlevel 1 goto erro
set "MABI_PY=.venv\Scripts\python.exe"

:executar
"%MABI_PY%" -c "import pandas, numpy, pyarrow, scipy, sklearn, xgboost, optuna, joblib, starlette, uvicorn, multipart" >nul 2>&1
if errorlevel 1 (
  "%MABI_PY%" -m pip install -r requirements.txt
  if errorlevel 1 goto erro
)
if not defined MABIYAMA_PORTA set "MABIYAMA_PORTA=8000"
echo Abra http://127.0.0.1:%MABIYAMA_PORTA% no navegador.
"%MABI_PY%" servidor.py
if errorlevel 1 goto erro
exit /b 0

:erro
echo Nao foi possivel iniciar o painel. Veja a mensagem acima.
pause
exit /b 1
