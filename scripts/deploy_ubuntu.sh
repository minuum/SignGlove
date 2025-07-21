#!/bin/bash

# SignGlove 우분투 서버 배포 스크립트
# 이 스크립트는 우분투 서버에 SignGlove 서버를 배포합니다.

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 변수 설정
PROJECT_NAME="signglove"
DEPLOY_DIR="/opt/signglove"
SERVICE_NAME="signglove.service"
USER_NAME="signglove"
BACKUP_DIR="/opt/signglove/backup"
DATA_DIR="/opt/signglove/data"
LOG_DIR="/var/log/signglove"

log_info "🚀 SignGlove 우분투 서버 배포 시작"

# 1. 시스템 요구사항 확인
log_info "📋 시스템 요구사항 확인"

# Ubuntu 버전 확인
if ! lsb_release -d | grep -q "Ubuntu"; then
    log_error "우분투 시스템이 아닙니다. 이 스크립트는 우분투 전용입니다."
    exit 1
fi

# Docker 설치 확인
if ! command -v docker &> /dev/null; then
    log_warning "Docker가 설치되지 않았습니다. Docker를 설치하시겠습니까? (y/n)"
    read -r install_docker
    if [[ $install_docker == "y" || $install_docker == "Y" ]]; then
        log_info "Docker 설치 중..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
        log_success "Docker 설치 완료"
    else
        log_error "Docker가 필요합니다. 배포를 중단합니다."
        exit 1
    fi
fi

# Docker Compose 설치 확인
if ! command -v docker-compose &> /dev/null; then
    log_info "Docker Compose 설치 중..."
    sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    log_success "Docker Compose 설치 완료"
fi

# 2. 사용자 및 디렉토리 생성
log_info "👤 사용자 및 디렉토리 설정"

if ! id "$USER_NAME" &>/dev/null; then
    log_info "signglove 사용자 생성"
    sudo useradd -r -s /bin/false -d $DEPLOY_DIR $USER_NAME
else
    log_info "signglove 사용자 이미 존재"
fi

# 디렉토리 생성
sudo mkdir -p $DEPLOY_DIR $DATA_DIR $BACKUP_DIR $LOG_DIR
sudo mkdir -p $DEPLOY_DIR/{config,scripts,deploy}

# 권한 설정
sudo chown -R $USER_NAME:$USER_NAME $DEPLOY_DIR $DATA_DIR $BACKUP_DIR $LOG_DIR
sudo chmod 755 $DEPLOY_DIR
sudo chmod 750 $DATA_DIR $BACKUP_DIR
sudo chmod 755 $LOG_DIR

log_success "사용자 및 디렉토리 설정 완료"

# 3. 프로젝트 파일 복사
log_info "📁 프로젝트 파일 복사"

# 현재 디렉토리가 프로젝트 루트인지 확인
if [[ ! -f "pyproject.toml" ]]; then
    log_error "프로젝트 루트 디렉토리에서 실행해주세요."
    exit 1
fi

# 기존 파일 백업 (있는 경우)
if [[ -d "$DEPLOY_DIR/server" ]]; then
    log_info "기존 배포 백업 중..."
    sudo tar -czf "$BACKUP_DIR/deploy_backup_$(date +%Y%m%d_%H%M%S).tar.gz" -C $DEPLOY_DIR . 2>/dev/null || true
fi

# 파일 복사
log_info "프로젝트 파일 복사 중..."
sudo cp -r . $DEPLOY_DIR/
sudo chown -R $USER_NAME:$USER_NAME $DEPLOY_DIR

# 실행 권한 설정
sudo chmod +x $DEPLOY_DIR/scripts/*.sh 2>/dev/null || true
sudo chmod +x $DEPLOY_DIR/deploy/entrypoint.sh

log_success "프로젝트 파일 복사 완료"

# 4. 환경 설정
log_info "⚙️ 환경 설정"

# 환경변수 파일 설정
if [[ ! -f "$DEPLOY_DIR/config/.env" ]]; then
    log_info "환경변수 파일 생성"
    sudo cp $DEPLOY_DIR/config/ubuntu.env.example $DEPLOY_DIR/config/.env
    
    # 기본값 설정
    sudo sed -i "s/your_secret_key_here_change_in_production/$(openssl rand -hex 32)/" $DEPLOY_DIR/config/.env
    sudo sed -i "s/your_wifi_password/SignGlove2025/" $DEPLOY_DIR/config/.env
    
    log_warning "환경변수 파일을 확인하고 필요한 값들을 설정해주세요: $DEPLOY_DIR/config/.env"
else
    log_info "환경변수 파일 이미 존재"
fi

sudo chown $USER_NAME:$USER_NAME $DEPLOY_DIR/config/.env
sudo chmod 600 $DEPLOY_DIR/config/.env

# 5. 시스템 서비스 설치
log_info "🔧 시스템 서비스 설치"

sudo cp $DEPLOY_DIR/deploy/signglove.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable signglove.service

log_success "시스템 서비스 설치 완료"

# 6. Docker 이미지 빌드
log_info "🐳 Docker 이미지 빌드"

cd $DEPLOY_DIR
sudo docker-compose build --no-cache

log_success "Docker 이미지 빌드 완료"

# 7. 서비스 시작
log_info "🚀 서비스 시작"

sudo systemctl start signglove.service

# 서비스 상태 확인
sleep 10
if sudo systemctl is-active --quiet signglove.service; then
    log_success "SignGlove 서비스 시작 완료!"
    
    # 헬스체크
    log_info "헬스체크 수행 중..."
    for i in {1..30}; do
        if curl -f http://localhost:8000/health &>/dev/null; then
            log_success "서버가 정상적으로 실행 중입니다!"
            break
        fi
        if [[ $i -eq 30 ]]; then
            log_warning "서버 응답 대기 중... (30초 경과)"
        fi
        sleep 2
    done
else
    log_error "SignGlove 서비스 시작 실패"
    sudo systemctl status signglove.service
    exit 1
fi

# 8. 최종 정보 출력
log_success "🎉 SignGlove 배포 완료!"
echo ""
echo "=== 배포 정보 ==="
echo "배포 디렉토리: $DEPLOY_DIR"
echo "데이터 디렉토리: $DATA_DIR"
echo "백업 디렉토리: $BACKUP_DIR"
echo "로그 디렉토리: $LOG_DIR"
echo ""
echo "=== 서비스 관리 명령어 ==="
echo "상태 확인: sudo systemctl status signglove"
echo "로그 확인: sudo journalctl -u signglove -f"
echo "재시작:   sudo systemctl restart signglove"
echo "중지:     sudo systemctl stop signglove"
echo ""
echo "=== 서버 접속 정보 ==="
echo "서버 URL: http://$(hostname -I | awk '{print $1}'):8000"
echo "헬스체크: curl http://localhost:8000/health"
echo "API 문서: http://$(hostname -I | awk '{print $1}'):8000/docs"
echo ""
echo "=== 다음 단계 ==="
echo "1. 환경변수 파일 확인: $DEPLOY_DIR/config/.env"
echo "2. 네트워크 방화벽 설정 (포트 8000 허용)"
echo "3. SSL 인증서 설정 (프로덕션 환경)"
echo "4. 모니터링 설정 확인"
echo ""

log_success "배포 스크립트 완료! 🚀" 