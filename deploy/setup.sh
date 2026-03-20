#!/bin/bash
set -e

APP_DIR="/root/SearchBarbara"

echo "=== [1/5] 安装系统依赖 ==="
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip > /dev/null

echo "=== [2/5] 创建 Python 虚拟环境 ==="
cd "$APP_DIR"
python3 -m venv .venv
source .venv/bin/activate
pip install --quiet -r requirements.txt

echo "=== [3/5] 检查 .env 配置 ==="
if [ ! -f "$APP_DIR/.env" ]; then
    echo "ERROR: 请先创建 $APP_DIR/.env 文件"
    echo "可参考 deploy/.env.example 模板"
    exit 1
fi
echo ".env 已就绪"

echo "=== [4/5] 安装 systemd 服务 ==="
cp "$APP_DIR/deploy/systemd/searchbarbara.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable searchbarbara

echo "=== [5/5] 启动服务 ==="
systemctl restart searchbarbara
sleep 2
systemctl status searchbarbara --no-pager

echo ""
echo "=== 部署完成 ==="
echo "访问地址: http://170.106.177.52:8000"
echo "查看日志: journalctl -u searchbarbara -f"
echo ""
echo "提醒: 请确保腾讯云安全组已放行 8000 端口(TCP)"
