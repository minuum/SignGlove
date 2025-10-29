# External Repositories

이 폴더는 SignGlove 프로젝트에서 사용하는 외부 저장소들을 Git Submodule로 관리합니다.

## 📁 관리 중인 저장소

- **KLP-SignGlove**: 메인 AI 모델 및 추론 시스템
- **SignGlove_HW**: 하드웨어 데이터 수집 및 처리
- **SignGlove-DataAnalysis**: 데이터 분석 및 시각화 도구

## 🚀 빠른 시작

### 1. 서브모듈 설정 (최초 1회)
```bash
./external/setup_submodules.sh
```

### 2. 서브모듈 업데이트
```bash
./external/update_submodules.sh
```

### 3. 수동 업데이트
```bash
git submodule update --remote --merge
```

## 📚 상세 가이드

자세한 사용법은 [SUBMODULE_MANAGEMENT.md](./SUBMODULE_MANAGEMENT.md)를 참고하세요.

## ⚠️ 주의사항

- 기존 폴더의 내용은 보존됩니다
- 서브모듈 작업 전에 반드시 백업을 만드세요
- 서브모듈 수정 시 원본 저장소에 영향을 줄 수 있습니다
