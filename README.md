# SignGlove - 통합 수어 인식 시스템

## 🎯 **프로젝트 개요**

SignGlove는 한국어 수어 인식을 위한 통합 시스템입니다. 이 저장소는 전체 프로젝트의 **메인 레포지토리**로서, 외부 팀들의 구현체들을 통합하고 관리하는 역할을 합니다.

## 📁 **프로젝트 구조**

```
SignGlove/ (메인 레포지토리)
├── src/                    # 핵심 통합 코드
│   ├── core/              # 핵심 기능
│   ├── integration/       # 외부 시스템 통합
│   └── deployment/        # 배포 관련
├── external/              # 외부 의존성 (자동 동기화)
│   ├── KLP-SignGlove/     # 딥러닝 모델 (Kyle-Riss 팀)
│   └── SignGlove_HW/      # 하드웨어 구현 (KNDG01001 팀)
├── docs/                  # 문서
│   ├── api/               # API 문서
│   ├── guides/            # 사용 가이드
│   └── architecture/      # 아키텍처 문서
├── config/                # 설정 파일
├── deploy/                # 배포 스크립트
├── tests/                 # 통합 테스트
└── dependencies/          # 의존성 관리
```

## 🔗 **외부 의존성**

### KLP-SignGlove (딥러닝 모델)
- **저장소**: https://github.com/Kyle-Riss/KLP-SignGlove
- **역할**: 한국어 수어 인식 딥러닝 모델
- **주요 기능**: 데이터 전처리, 모델 학습, 추론

### SignGlove_HW (하드웨어)
- **저장소**: https://github.com/KNDG01001/SignGlove_HW
- **역할**: 센서 데이터 수집 및 처리
- **주요 기능**: IMU 센서 처리, 데이터 수집, 하드웨어 통신

## 🚀 **시작하기**

### 1. 저장소 클론
```bash
git clone https://github.com/minuum/SignGlove.git
cd SignGlove
```

### 2. 외부 의존성 동기화
```bash
# 자동 동기화 (GitHub Actions)
# 또는 수동 동기화
cd external/KLP-SignGlove
git clone https://github.com/Kyle-Riss/KLP-SignGlove.git temp
cp -r temp/* .
rm -rf temp

cd ../SignGlove_HW
git clone https://github.com/KNDG01001/SignGlove_HW.git temp
cp -r temp/* .
rm -rf temp
```

### 3. 환경 설정
```bash
# Python 환경 설정
poetry install

# 또는 pip 사용
pip install -r dependencies/requirements.txt
```

## 🔧 **주요 기능**

### 통합 시스템
- 외부 팀 구현체들의 통합 관리
- API 게이트웨이 및 라우팅
- 데이터 파이프라인 관리
- 배포 및 운영 관리

### 자동화
- GitHub Actions를 통한 자동 동기화
- CI/CD 파이프라인
- 자동 테스트 및 배포

## 📚 **문서**

- [API 문서](docs/api/)
- [사용 가이드](docs/guides/)
- [아키텍처 문서](docs/architecture/)
- [외부 의존성 관리](dependencies/external-repos.md)

## 🤝 **기여하기**

1. 이슈 생성 또는 기존 이슈 확인
2. 브랜치 생성 (`feature/기능명` 또는 `fix/버그명`)
3. 코드 작성 및 테스트
4. Pull Request 생성

## 📋 **개발 상태**

- [x] 프로젝트 구조 설정
- [x] 외부 의존성 통합
- [x] 자동 동기화 설정
- [ ] API 게이트웨이 구현
- [ ] 통합 테스트 작성
- [ ] 배포 파이프라인 구축

## 📞 **연락처**

- **프로젝트 관리자**: minuum
- **KLP-SignGlove 팀**: Kyle-Riss
- **SignGlove_HW 팀**: KNDG01001

## 📄 **라이선스**

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 외부 의존성들은 각각의 라이선스를 따릅니다. 