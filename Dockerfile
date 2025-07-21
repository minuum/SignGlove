# SignGlove Ubuntu 서버용 Dockerfile
FROM python:3.11-slim

# 우분투 패키지 업데이트 및 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    nano \
    htop \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /opt/signglove

# Poetry 설치
RUN pip install poetry==1.6.1

# Poetry 설정 (가상환경을 컨테이너 내부에 생성하지 않음)
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# 의존성 파일 복사 및 설치
COPY pyproject.toml poetry.lock ./
RUN poetry install --only=main --no-root && rm -rf $POETRY_CACHE_DIR

# 애플리케이션 코드 복사
COPY . .

# 권한 설정
RUN chmod +x /opt/signglove/scripts/*.py

# 데이터 디렉토리 생성
RUN mkdir -p /opt/signglove/data \
    /opt/signglove/backup \
    /var/log/signglove \
    /opt/signglove/config

# 환경변수 파일 복사
COPY config/ubuntu.env.example /opt/signglove/config/.env

# 포트 노출
EXPOSE 8000 9090

# Supervisor 설정 파일 복사
COPY deploy/supervisord.conf /etc/supervisor/conf.d/signglove.conf

# 헬스체크 스크립트 추가
COPY deploy/healthcheck.py /usr/local/bin/healthcheck.py
RUN chmod +x /usr/local/bin/healthcheck.py

# 헬스체크 설정
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /usr/local/bin/healthcheck.py

# 시작 스크립트
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 볼륨 마운트 포인트
VOLUME ["/opt/signglove/data", "/opt/signglove/backup", "/var/log/signglove"]

# 환경변수 설정
ENV PYTHONPATH=/opt/signglove \
    ENVIRONMENT=production \
    HOST=0.0.0.0 \
    PORT=8000

# 비루트 사용자 생성 및 권한 설정
RUN adduser --disabled-password --gecos '' signglove && \
    chown -R signglove:signglove /opt/signglove /var/log/signglove
USER signglove

# 시작점
ENTRYPOINT ["/entrypoint.sh"]
CMD ["poetry", "run", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"] 