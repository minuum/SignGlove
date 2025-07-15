# 🚀 SignGlove 빠른 시작 가이드

## 1분 만에 시작하기

### 1. 환경 설정
```bash
# 프로젝트 클론
git clone <repository-url>
cd SignGlove

# Poetry 의존성 설치
poetry install

# 가상환경 활성화
poetry shell
```

### 2. 전체 데모 실행 (추천)
```bash
python scripts/demo.py
```

이 명령어 하나로 다음이 모두 실행됩니다:
- ✅ 서버 자동 시작
- ✅ 연결성 테스트
- ✅ 센서 데이터 테스트
- ✅ 제스처 데이터 테스트
- ✅ 성능 테스트
- ✅ 더미 데이터 생성
- ✅ 자동 서버 종료

## 단계별 실행

### 1단계: 서버 시작
```bash
python scripts/start_server.py
```

### 2단계: 테스트 실행 (새 터미널)
```bash
python scripts/run_tests.py
```

### 3단계: 더미 데이터 생성 (새 터미널)
```bash
python tests/dummy_data_generator.py
```

## 테스트 결과 확인

### 웹 브라우저
- API 문서: http://localhost:8000/docs
- 서버 상태: http://localhost:8000/

### 생성된 데이터 파일
```
data/raw/
├── sensor_data_20240103.csv      # 센서 데이터
├── gesture_data_20240103.csv     # 제스처 메타데이터
└── gesture_sequences_20240103.json # 제스처 시퀀스
```

## 문제 해결

### 서버 시작 안됨
```bash
# 포트 충돌 해결
lsof -ti:8000 | xargs kill -9

# 의존성 재설치
poetry install --no-cache
```

### 테스트 실패
```bash
# 서버 상태 확인
curl http://localhost:8000/health

# 로그 확인
tail -f server.log
```

## 다음 단계

1. **하드웨어 연결**: `hardware/arduino/sign_glove_client.ino` 설정
2. **데이터 분석**: `data/raw/` 폴더의 CSV 파일 분석
3. **클래스 정의**: 팀원과 협업하여 수어 클래스 정의
4. **성능 최적화**: 실제 환경에서 성능 튜닝

🎉 **축하합니다!** SignGlove 시스템이 성공적으로 실행되고 있습니다. 