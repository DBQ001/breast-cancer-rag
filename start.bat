@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "APP_URL=http://127.0.0.1:5000"

echo ==============================================================
echo    乳愈智能 · 乳腺癌 AI 预测系统   ——   一键启动
echo ==============================================================
echo.

REM ---------- 1. 查找可用的 Python 解释器 ----------
set "PY="
call :try_py "py" "-3"
call :try_py "python3"
call :try_py "python"
call :try_py "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"

if not defined PY (
    echo [错误] 未找到可用的 Python 解释器。
    echo.
    echo        请先安装 Python 3.9 或更高版本：
    echo        https://www.python.org/downloads/
    echo        安装时请务必勾选 "Add python.exe to PATH"。
    echo.
    pause
    exit /b 1
)
echo [1/3] Python 解释器： %PY%

REM ---------- 2. 检查依赖，缺失则自动安装 ----------
%PY% -c "import flask, sklearn, numpy, pandas, joblib, requests" >nul 2>nul
if errorlevel 1 (
    echo [2/3] 首次运行，正在安装依赖，请稍候...
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [错误] 依赖安装失败，请检查网络后重试。
        echo        也可手动执行： %PY% -m pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
) else (
    echo [2/3] 依赖检查通过
)

REM ---------- 3. 启动服务并自动打开浏览器 ----------
echo [3/3] 正在启动服务： %APP_URL%
echo.
echo       浏览器将在 3 秒后自动打开
echo       如需停止服务，请按 Ctrl + C
echo.

REM 延迟 3 秒再打开浏览器，避免服务尚未就绪
start "" cmd /c "timeout /t 3 /nobreak >nul & start %APP_URL%"

%PY% app.py

echo.
echo 服务已停止，感谢使用。
pause
exit /b 0

REM ==============================================================
REM 子过程：依次测试候选解释器，把第一个可用的写入 PY
REM ==============================================================
:try_py
if defined PY exit /b 0
"%~1" %~2 %~3 -c "import sys" >nul 2>nul
if not errorlevel 1 set "PY=%~1 %~2 %~3"
exit /b 0
