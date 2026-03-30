@echo off
echo ========================================
echo H2Former 后端服务启动脚本
echo =====================================
echo.

REM 检查Python是否可用
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python，请先安装Python 3.7+
    pause
    exit /b 1
)

REM 检查虚拟环境
if not exist venv (
    echo 创建虚拟环境...
    python -m venv venv
)

echo 激活虚拟环境...
call venv\Scripts\activate.bat

echo 安装依赖...
pip install -r requirements.txt

echo.
echo ========================================
echo 启动Flask后端服务...
echo 服务地址: http://localhost:5000
echo API文档: http://localhost:5000/api/health
echo ========================================
echo.

REM 启动Flask服务
python app.py

echo.
pause
