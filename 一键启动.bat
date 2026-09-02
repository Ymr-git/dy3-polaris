@echo off
title Dy3+ Polaris 一键启动
setlocal enabledelayedexpansion

REM ============================================================
REM  Dy3+ Polaris 科研证据分析与多智能体协同决策系统 —— 一键启动
REM  本脚本位于代码仓库根目录（04-编码），不依赖写死路径，可随仓库移动。
REM  自动完成：探测 Python → 装依赖 → 启动服务 → 打开浏览器
REM ============================================================

REM ---- 1. 定位项目目录（以脚本自身位置为准）----
set "CODE=%~dp0"

REM ---- 2. 自动探测 Python（优先本地虚拟环境，其次 PATH，再次 py 启动器）----
set "PY="
if exist "%CODE%.venv\Scripts\python.exe" set "PY=%CODE%.venv\Scripts\python.exe"
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    where py >nul 2>nul
    if not errorlevel 1 set "PY=py"
)
if not defined PY (
    echo.
    echo [错误] 未找到 Python。
    echo        请先安装 Python 3.10 或更高版本，
    echo        安装时务必勾选 "Add Python to PATH"。
    echo.
    pause
    exit /b 1
)

echo 正在使用 Python: %PY%
"%PY%" --version

REM ---- 3. 进入代码目录 ----
cd /d "%CODE%"

REM ---- 4. 首次运行自动安装依赖 ----
"%PY%" -c "import dy3_polaris" >nul 2>nul
if errorlevel 1 (
    echo.
    echo 首次运行，正在自动安装依赖（约 1-3 分钟，请耐心等待）...
    echo.
    "%PY%" -m pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo.
        echo [错误] 依赖安装失败，请检查网络后重试。
        echo        也可手动执行: "%PY%" -m pip install -e ".[dev]"
        echo.
        pause
        exit /b 1
    )
)

REM ---- 5. 若无 .env 则从模板创建 ----
if not exist "%CODE%.env" (
    echo.
    echo 首次运行，正在从 .env.example 创建本地配置 .env ...
    copy /y "%CODE%.env.example" "%CODE%.env" >nul
)

REM ---- 6. 启动服务 ----
echo.
echo 正在启动 Dy3+ Polaris ...
echo 启动完成后浏览器将自动打开 http://127.0.0.1:8000/
echo 关闭本窗口即可停止服务。
echo.

start "" "http://127.0.0.1:8000/"
"%PY%" -m dy3_polaris.main --host 127.0.0.1 --port 8000

pause
