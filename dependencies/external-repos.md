# External Dependencies

이 문서는 SignGlove 프로젝트에서 사용하는 외부 저장소들의 정보를 관리합니다.

## 📚 **외부 저장소 목록**

### 1. KLP-SignGlove
- **저장소**: https://github.com/Kyle-Riss/KLP-SignGlove
- **소유자**: Kyle-Riss
- **설명**: 한국어 수어 인식을 위한 딥러닝 모델 및 데이터 처리 코드
- **사용 목적**: 수어 인식 모델 참조 및 학습
- **동기화 주기**: 매주 월요일
- **마지막 업데이트**: 2024-08-28

### 2. SignGlove_HW
- **저장소**: https://github.com/KNDG01001/SignGlove_HW
- **소유자**: KNDG01001
- **설명**: 하드웨어 센서 데이터 수집 및 처리 코드
- **사용 목적**: 센서 데이터 처리 방법 참조
- **동기화 주기**: 매주 월요일
- **마지막 업데이트**: 2024-08-28

### 3. SignGlove-DataAnalysis
- **저장소**: https://github.com/wodu2s/SignGlove-DataAnalysis
- **소유자**: wodu2s
- **설명**: 센서 데이터 분석 및 시각화 도구
- **사용 목적**: 데이터 품질 분석 및 전처리 방법 참조
- **동기화 주기**: 매주 월요일
- **마지막 업데이트**: 2024-12-19

## 🔄 **동기화 방법**

### 자동 동기화 (권장)
GitHub Actions가 매주 자동으로 외부 저장소를 동기화합니다.

### 수동 동기화
```bash
# KLP-SignGlove 업데이트
cd collaborations/KLP-SignGlove
git pull origin main

# SignGlove_HW 업데이트
cd ../SignGlove_HW
git pull origin main

# SignGlove-DataAnalysis 업데이트
cd ../SignGlove-DataAnalysis
git pull origin main
```

## 📋 **의존성 관리**

### Python 패키지
- `pyproject.toml`에서 관리
- Poetry를 사용한 의존성 관리

### 외부 저장소
- 이 문서에서 버전 정보 관리
- GitHub Actions로 자동 동기화
- `collaborations/` 폴더에 저장

## 🚨 **주의사항**

1. **저작권**: 외부 저장소의 코드는 각각의 라이선스를 따릅니다.
2. **업데이트**: 외부 저장소 업데이트 시 호환성 확인 필요
3. **백업**: 중요한 변경사항은 별도로 백업
4. **테스트**: 외부 코드 업데이트 후 테스트 필수

## 📞 **연락처**

외부 저장소 관련 문의사항이 있으면 다음으로 연락하세요:
- KLP-SignGlove: Kyle-Riss
- SignGlove_HW: KNDG01001
- SignGlove-DataAnalysis: wodu2s
