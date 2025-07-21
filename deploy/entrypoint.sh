#!/bin/bash

# SignGlove Ubuntu 서버 시작 스크립트
# 컨테이너 시작 시 실행되는 초기화 스크립트

set -e

echo "🚀 SignGlove 서버 시작 중..."

# 환경변수 설정 확인
if [ -f /opt/signglove/config/.env ]; then
    echo "✅ 환경변수 파일 로드"
    source /opt/signglove/config/.env
else
    echo "⚠️ 환경변수 파일 없음 - 기본값 사용"
fi

# 데이터 디렉토리 권한 확인
echo "📁 데이터 디렉토리 권한 설정"
mkdir -p /opt/signglove/data/sensor_data \
         /opt/signglove/data/gesture_data \
         /opt/signglove/backup \
         /var/log/signglove

# Poetry 가상환경 활성화
export PATH="/opt/signglove/.venv/bin:$PATH"

# 데이터베이스 초기화 (필요시)
echo "🗃️ 데이터베이스 초기화"
if [ ! -f /opt/signglove/data/signglove.db ]; then
    echo "새로운 데이터베이스 생성"
    # TODO: 데이터베이스 초기화 스크립트 실행
fi

# 서버 상태 확인
echo "🔍 서버 상태 사전 점검"
poetry run python -c "
import sys
sys.path.append('/opt/signglove')
try:
    from server.main import app
    from server.data_storage import DataStorage
    from server.professor_proxy import professor_proxy
    print('✅ 모든 모듈 임포트 성공')
except Exception as e:
    print(f'❌ 모듈 임포트 실패: {e}')
    sys.exit(1)
"

# 교수님 서버 연결 테스트 (선택사항)
if [ "$ENABLE_PROFESSOR_PROXY" = "true" ]; then
    echo "🔗 교수님 서버 연결 테스트"
    poetry run python -c "
import asyncio
import sys
sys.path.append('/opt/signglove')
from server.professor_proxy import professor_proxy

async def test_connection():
    status = await professor_proxy.get_server_status()
    if status['status'] == 'connected':
        print('✅ 교수님 서버 연결 성공')
    else:
        print(f'⚠️ 교수님 서버 연결 실패: {status[\"message\"]}')

asyncio.run(test_connection())
    "
fi

# KSL 클래스 초기화
echo "📚 KSL 클래스 초기화"
poetry run python -c "
import sys
sys.path.append('/opt/signglove')
from server.ksl_classes import ksl_manager
stats = ksl_manager.get_statistics()
print(f'✅ KSL 클래스 {stats[\"total_classes\"]}개 로드됨')
print(f'   - 모음: {stats[\"by_category\"][\"vowel\"]}개')
print(f'   - 자음: {stats[\"by_category\"][\"consonant\"]}개')
print(f'   - 숫자: {stats[\"by_category\"][\"number\"]}개')
"

# 로그 파일 초기화
echo "📝 로그 시스템 초기화"
touch /var/log/signglove/server.log
touch /var/log/signglove/error.log
touch /var/log/signglove/access.log

# 시스템 정보 출력
echo "💻 시스템 정보:"
echo "   - Python 버전: $(python --version)"
echo "   - Poetry 버전: $(poetry --version)"
echo "   - 작업 디렉토리: $(pwd)"
echo "   - 환경: ${ENVIRONMENT:-development}"
echo "   - 호스트: ${HOST:-0.0.0.0}"
echo "   - 포트: ${PORT:-8000}"

# 네트워크 정보 출력
echo "🌐 네트워크 정보:"
if command -v hostname &> /dev/null; then
    echo "   - 호스트명: $(hostname)"
fi
if command -v ip &> /dev/null; then
    echo "   - IP 주소: $(ip addr show | grep 'inet ' | grep -v 127.0.0.1 | awk '{print $2}' | head -1)"
fi

echo "✅ SignGlove 서버 초기화 완료!"
echo "🎯 서버 시작: $*"

# 인자로 받은 명령어 실행
exec "$@" 