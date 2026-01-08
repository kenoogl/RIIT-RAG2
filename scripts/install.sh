#!/bin/bash

# 玄界RAGシステム インストールスクリプト

set -e

# 色付きの出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ログ関数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 設定
INSTALL_DIR="/opt/genkai-rag-system"
SERVICE_USER="genkai"
PYTHON_VERSION="3.11"

echo "=========================================="
echo "玄界RAGシステム インストール"
echo "=========================================="

# rootユーザーかチェック
if [ "$EUID" -ne 0 ]; then
    log_error "This script must be run as root"
    exit 1
fi

# システムの更新
log_info "Updating system packages..."
apt-get update

# 必要なパッケージをインストール
log_info "Installing system dependencies..."
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    git \
    build-essential \
    nginx \
    supervisor

# Pythonのバージョンをチェック
PYTHON_CMD="python3"
PYTHON_ACTUAL_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)

log_info "Python version: $PYTHON_ACTUAL_VERSION"

# サービスユーザーを作成
if ! id "$SERVICE_USER" &>/dev/null; then
    log_info "Creating service user: $SERVICE_USER"
    useradd -r -s /bin/false -d "$INSTALL_DIR" "$SERVICE_USER"
else
    log_info "Service user $SERVICE_USER already exists"
fi

# インストールディレクトリを作成
log_info "Creating installation directory: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

# 現在のディレクトリからファイルをコピー
log_info "Copying application files..."
cp -r . "$INSTALL_DIR/"

# 権限を設定
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

# 仮想環境を作成
log_info "Creating Python virtual environment..."
sudo -u "$SERVICE_USER" $PYTHON_CMD -m venv "$INSTALL_DIR/venv"

# 依存関係をインストール
log_info "Installing Python dependencies..."
sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install --upgrade pip
sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"

# 設定ファイルを準備
log_info "Setting up configuration files..."
if [ ! -f "$INSTALL_DIR/.env" ]; then
    cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
    log_info "Created .env file from example. Please edit it with your settings."
fi

# ディレクトリを作成
sudo -u "$SERVICE_USER" mkdir -p "$INSTALL_DIR/data"
sudo -u "$SERVICE_USER" mkdir -p "$INSTALL_DIR/logs"
sudo -u "$SERVICE_USER" mkdir -p "$INSTALL_DIR/backups"

# systemdサービスをインストール
log_info "Installing systemd service..."
cp "$INSTALL_DIR/scripts/genkai-rag.service" /etc/systemd/system/
systemctl daemon-reload

# Nginxの設定（オプション）
if command -v nginx &> /dev/null; then
    log_info "Setting up Nginx reverse proxy..."
    cat > /etc/nginx/sites-available/genkai-rag << EOF
server {
    listen 80;
    server_name localhost;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
EOF

    # サイトを有効化
    ln -sf /etc/nginx/sites-available/genkai-rag /etc/nginx/sites-enabled/
    nginx -t && systemctl reload nginx
    log_info "Nginx configuration completed"
fi

# ファイアウォール設定（UFWが利用可能な場合）
if command -v ufw &> /dev/null; then
    log_info "Configuring firewall..."
    ufw allow 8000/tcp
    ufw allow 80/tcp
    ufw allow 443/tcp
fi

# サービスを有効化
log_info "Enabling Genkai RAG System service..."
systemctl enable genkai-rag

echo "=========================================="
log_info "Installation completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit configuration: $INSTALL_DIR/.env"
echo "2. Start the service: systemctl start genkai-rag"
echo "3. Check status: systemctl status genkai-rag"
echo "4. View logs: journalctl -u genkai-rag -f"
echo ""
echo "Web interface will be available at:"
echo "- http://localhost:8000 (direct)"
echo "- http://localhost (via Nginx, if configured)"
echo ""
log_warn "Don't forget to:"
log_warn "- Install and configure Ollama"
log_warn "- Download required LLM models"
log_warn "- Update the .env file with your settings"