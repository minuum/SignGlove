# SignGlove 우분투 서버 배포 가이드

## 📋 개요

이 문서는 SignGlove 데이터 수집 서버를 우분투 서버에 배포하는 방법을 안내합니다.

**날짜**: 2025.07.15  
**대상**: 우분투 18.04+ 서버  
**배포 방식**: Docker + Docker Compose + systemd 서비스

## 🔧 시스템 요구사항

### 최소 사양
- **OS**: Ubuntu 18.04 LTS 이상
- **CPU**: 2 코어 이상
- **RAM**: 4GB 이상
- **Storage**: 20GB 이상
- **네트워크**: 인터넷 연결 필수

### 권장 사양
- **OS**: Ubuntu 22.04 LTS
- **CPU**: 4 코어 이상
- **RAM**: 8GB 이상  
- **Storage**: 50GB 이상 (SSD 권장)
- **네트워크**: 1Gbps 이상

## 🚀 빠른 배포 (자동 스크립트)

### 1단계: 저장소 클론
```bash
git clone https://github.com/SignGlove/server.git
cd server
```

### 2단계: 배포 스크립트 실행
```bash
chmod +x scripts/deploy_ubuntu.sh
sudo ./scripts/deploy_ubuntu.sh
```

### 3단계: 서버 확인
```bash
curl http://localhost:8000/health
```

**완료!** 서버가 정상적으로 실행 중입니다.

## 📝 수동 배포 (단계별)

자동 스크립트가 작동하지 않는 경우 수동으로 배포할 수 있습니다.

### 1. Docker 설치
```bash
# Docker 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Docker Compose 설치
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 재로그인 또는 새 그룹 적용
newgrp docker
```

### 2. 사용자 및 디렉토리 생성
```bash
# signglove 사용자 생성
sudo useradd -r -s /bin/false -d /opt/signglove signglove

# 필요한 디렉토리 생성
sudo mkdir -p /opt/signglove/{data,backup,config}
sudo mkdir -p /var/log/signglove

# 권한 설정
sudo chown -R signglove:signglove /opt/signglove /var/log/signglove
sudo chmod 755 /opt/signglove
sudo chmod 750 /opt/signglove/data /opt/signglove/backup
```

### 3. 프로젝트 파일 복사
```bash
# 프로젝트 파일을 /opt/signglove로 복사
sudo cp -r . /opt/signglove/
sudo chown -R signglove:signglove /opt/signglove
sudo chmod +x /opt/signglove/scripts/*.sh
sudo chmod +x /opt/signglove/deploy/entrypoint.sh
```

### 4. 환경 설정
```bash
# 환경변수 파일 생성
sudo cp /opt/signglove/config/ubuntu.env.example /opt/signglove/config/.env

# 보안 키 생성
SECRET_KEY=$(openssl rand -hex 32)
sudo sed -i "s/your_secret_key_here_change_in_production/$SECRET_KEY/" /opt/signglove/config/.env

# 환경변수 파일 권한 설정
sudo chown signglove:signglove /opt/signglove/config/.env
sudo chmod 600 /opt/signglove/config/.env
```

### 5. 시스템 서비스 등록
```bash
# 서비스 파일 복사
sudo cp /opt/signglove/deploy/signglove.service /etc/systemd/system/

# 서비스 활성화
sudo systemctl daemon-reload
sudo systemctl enable signglove.service
```

### 6. Docker 이미지 빌드 및 실행
```bash
cd /opt/signglove

# Docker 이미지 빌드
sudo docker-compose build

# 서비스 시작
sudo systemctl start signglove.service
```

## ⚙️ 환경 설정

### 중요 환경변수 설정

`/opt/signglove/config/.env` 파일을 수정하여 다음 값들을 설정하세요:

```bash
# 서버 설정
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=production

# 보안 설정 (반드시 변경 필요)
SECRET_KEY=your_unique_secret_key_here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# 교수님 서버 연동 (필요시)
PROFESSOR_SERVER_URL=http://professor-server:8080/api
PROFESSOR_SERVER_TOKEN=your_professor_server_token
ENABLE_PROFESSOR_PROXY=false

# Arduino 하드웨어 설정
ARDUINO_WIFI_SSID=SignGlove_Network
ARDUINO_WIFI_PASSWORD=your_secure_wifi_password

# 데이터베이스 (향후 PostgreSQL 사용시)
DATABASE_URL=sqlite:///opt/signglove/data/signglove.db
```

## 🔍 서비스 관리

### 기본 명령어
```bash
# 서비스 상태 확인
sudo systemctl status signglove

# 서비스 시작/중지/재시작
sudo systemctl start signglove
sudo systemctl stop signglove
sudo systemctl restart signglove

# 로그 확인
sudo journalctl -u signglove -f

# Docker 컨테이너 상태 확인
sudo docker-compose -f /opt/signglove/docker-compose.yml ps
```

### 로그 파일 위치
- **시스템 로그**: `sudo journalctl -u signglove`
- **애플리케이션 로그**: `/var/log/signglove/server.log`
- **에러 로그**: `/var/log/signglove/error.log`
- **Docker 로그**: `sudo docker-compose logs -f`

## 🌐 네트워크 설정

### 방화벽 설정 (ufw)
```bash
# 기본 정책 설정
sudo ufw default deny incoming
sudo ufw default allow outgoing

# SSH 허용 (원격 접속용)
sudo ufw allow ssh

# SignGlove 서버 포트 허용
sudo ufw allow 8000/tcp

# 방화벽 활성화
sudo ufw enable
```

### 네트워크 확인
```bash
# 서버 응답 확인
curl http://localhost:8000/health

# 네트워크 포트 확인
sudo netstat -tlnp | grep :8000

# 외부 접근 테스트 (다른 컴퓨터에서)
curl http://SERVER_IP:8000/health
```

## 📊 모니터링 및 헬스체크

### 내장 헬스체크
```bash
# 기본 헬스체크
curl http://localhost:8000/health

# 상세 헬스체크 (JSON 형태)
python /usr/local/bin/healthcheck.py --json
```

### API 엔드포인트
- **서버 상태**: `GET /health`
- **데이터 통계**: `GET /data/stats`
- **KSL 통계**: `GET /api/ksl/statistics`
- **교수님 서버 상태**: `GET /api/professor/status`

## 🔄 업데이트 및 백업

### 서비스 업데이트
```bash
# 저장소에서 최신 코드 가져오기
cd /opt/signglove
sudo -u signglove git pull origin main

# Docker 이미지 재빌드
sudo docker-compose build --no-cache

# 서비스 재시작
sudo systemctl restart signglove
```

### 데이터 백업
```bash
# 수동 백업
sudo tar -czf /opt/signglove/backup/manual_backup_$(date +%Y%m%d_%H%M%S).tar.gz \
  -C /opt/signglove/data .

# 자동 백업 (cron으로 설정)
sudo crontab -e
# 매일 새벽 2시에 백업
0 2 * * * /opt/signglove/scripts/backup_data.py --auto
```

## 🔧 문제 해결

### 일반적인 문제들

#### 1. 서비스 시작 실패
```bash
# 로그 확인
sudo journalctl -u signglove -n 50

# Docker 컨테이너 상태 확인
sudo docker-compose -f /opt/signglove/docker-compose.yml ps

# 권한 확인
ls -la /opt/signglove/
```

#### 2. 포트 충돌
```bash
# 8000번 포트 사용 중인 프로세스 확인
sudo netstat -tlnp | grep :8000
sudo lsof -i :8000

# 다른 포트로 변경 (환경변수 파일 수정)
sudo nano /opt/signglove/config/.env
# PORT=8001 로 변경 후 재시작
```

#### 3. Docker 권한 문제
```bash
# 현재 사용자를 docker 그룹에 추가
sudo usermod -aG docker $USER

# 재로그인 또는
newgrp docker
```

#### 4. 메모리 부족
```bash
# 메모리 사용량 확인
free -h
docker stats

# 불필요한 Docker 이미지 정리
sudo docker system prune -a
```

### 로그 분석
```bash
# 실시간 로그 모니터링
sudo journalctl -u signglove -f

# 에러만 필터링
sudo journalctl -u signglove | grep -i error

# 특정 기간 로그
sudo journalctl -u signglove --since "2025-07-15 10:00:00"
```

## 🚀 성능 최적화

### Docker 리소스 제한
`docker-compose.yml`에서 리소스 제한 설정:

```yaml
services:
  signglove-app:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'  
          memory: 2G
```

### 시스템 튜닝
```bash
# 파일 디스크립터 제한 증가
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# TCP 성능 튜닝
echo "net.core.rmem_max = 268435456" | sudo tee -a /etc/sysctl.conf
echo "net.core.wmem_max = 268435456" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## 🔒 보안 강화

### 기본 보안 설정
```bash
# SSH 키 기반 인증 설정
# 패스워드 인증 비활성화
sudo nano /etc/ssh/sshd_config
# PasswordAuthentication no

# 자동 업데이트 설정
sudo apt install unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

### SSL/TLS 설정 (선택사항)
```bash
# Let's Encrypt 인증서 설치
sudo apt install certbot
sudo certbot --standalone -d your-domain.com

# Nginx 리버스 프록시 설정 (HTTPS)
# docker-compose.yml의 nginx 서비스 활성화
```

## 📞 지원 및 문의

### 문제 리포팅
1. **GitHub Issues**: https://github.com/SignGlove/server/issues
2. **로그 파일 첨부**: `/var/log/signglove/` 디렉토리
3. **시스템 정보 포함**: `lsb_release -a`, `docker --version`

### 연락처
- **기술 지원**: 이민우 (minwoo@signglove.com)
- **하드웨어 문의**: 양동건 
- **KSL 데이터 문의**: YUBEEN, 정재연

---

**마지막 업데이트**: 2025.07.15  
**버전**: 1.0.0  
**상태**: 우분투 배포 완료 ✅ 