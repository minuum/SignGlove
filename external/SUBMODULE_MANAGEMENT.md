# Git Submodule 관리 가이드

## 📋 관리 중인 외부 저장소 목록

현재 `external/` 폴더에서 관리하고 있는 외부 저장소들:

| 저장소명 | 경로 | 설명 | 상태 |
|---------|------|------|------|
| KLP-SignGlove | `external/KLP-SignGlove/` | 메인 AI 모델 및 추론 시스템 | 🔄 관리 중 |
| SignGlove_HW | `external/SignGlove_HW/` | 하드웨어 데이터 수집 및 처리 | 🔄 관리 중 |
| SignGlove-DataAnalysis | `external/SignGlove-DataAnalysis/` | 데이터 분석 및 시각화 도구 | 🔄 관리 중 |

## 🚀 서브모듈 초기화 및 설정

### 1. 기존 폴더를 서브모듈로 변환

```bash
# 1. 기존 폴더를 임시로 백업
mv external/KLP-SignGlove external/KLP-SignGlove_backup
mv external/SignGlove_HW external/SignGlove_HW_backup
mv external/SignGlove-DataAnalysis external/SignGlove-DataAnalysis_backup

# 2. 서브모듈로 추가
git submodule add https://github.com/Kyle-Riss/KLP-SignGlove.git external/KLP-SignGlove
git submodule add https://github.com/username/SignGlove_HW.git external/SignGlove_HW
git submodule add https://github.com/username/SignGlove-DataAnalysis.git external/SignGlove-DataAnalysis

# 3. 백업 폴더의 변경사항을 서브모듈에 병합 (필요시)
# 각 서브모듈 폴더에서 git add, git commit 수행
```

### 2. 서브모듈 초기화 (새로 클론한 경우)

```bash
# 서브모듈 초기화 및 업데이트
git submodule init
git submodule update

# 또는 한 번에
git submodule update --init --recursive
```

## ⚠️ 서브모듈화 시 주의사항

### 1. **절대 기존 폴더를 삭제하지 마세요**
- 기존 `external/` 폴더의 내용은 보존되어야 합니다
- 서브모듈로 변환하기 전에 반드시 백업을 만드세요

### 2. **Git 히스토리 보존**
- 기존 폴더의 Git 히스토리는 유지됩니다
- 서브모듈로 변환해도 커밋 히스토리는 그대로 유지됩니다

### 3. **의존성 관리**
- 각 서브모듈의 `requirements.txt`나 `package.json`을 확인하세요
- 메인 프로젝트의 의존성과 충돌하지 않는지 확인하세요

### 4. **브랜치 관리**
- 서브모듈은 특정 커밋에 고정됩니다
- `main` 브랜치가 아닌 특정 태그나 커밋을 사용할 수 있습니다

## 🔄 서브모듈 업데이트 및 갱신

### 1. 모든 서브모듈 업데이트

```bash
# 모든 서브모듈을 최신 상태로 업데이트
git submodule update --remote

# 또는 각 서브모듈을 개별적으로 업데이트
git submodule update --remote external/KLP-SignGlove
git submodule update --remote external/SignGlove_HW
git submodule update --remote external/SignGlove-DataAnalysis
```

### 2. 특정 서브모듈 업데이트

```bash
# 특정 서브모듈로 이동
cd external/KLP-SignGlove

# 원격 저장소에서 최신 변경사항 가져오기
git fetch origin

# 특정 브랜치로 체크아웃 (예: main)
git checkout main

# 최신 변경사항으로 업데이트
git pull origin main

# 메인 프로젝트로 돌아가서 변경사항 커밋
cd ../..
git add external/KLP-SignGlove
git commit -m "Update KLP-SignGlove submodule"
```

### 3. 서브모듈 상태 확인

```bash
# 서브모듈 상태 확인
git submodule status

# 서브모듈 변경사항 확인
git submodule foreach git status

# 서브모듈의 원격 브랜치 확인
git submodule foreach git branch -r
```

## 🛠️ 서브모듈 관리 명령어

### 기본 명령어

```bash
# 서브모듈 목록 확인
git submodule

# 서브모듈 상태 확인
git submodule status

# 서브모듈 초기화
git submodule init

# 서브모듈 업데이트
git submodule update

# 서브모듈 추가
git submodule add <repository-url> <path>

# 서브모듈 제거
git submodule deinit <path>
git rm <path>
```

### 고급 명령어

```bash
# 모든 서브모듈에서 명령어 실행
git submodule foreach 'git pull origin main'

# 서브모듈의 특정 브랜치 추적
git config submodule.external/KLP-SignGlove.branch main

# 서브모듈 업데이트 시 자동으로 최신 커밋 사용
git submodule update --remote --merge
```

## 📝 서브모듈 작업 워크플로우

### 1. 일상적인 업데이트

```bash
# 1. 메인 프로젝트 업데이트
git pull origin main

# 2. 서브모듈 업데이트
git submodule update --init --recursive

# 3. 서브모듈을 최신 상태로 업데이트
git submodule update --remote
```

### 2. 서브모듈 수정 및 커밋

```bash
# 1. 서브모듈로 이동
cd external/KLP-SignGlove

# 2. 수정 작업 수행
# ... 코드 수정 ...

# 3. 서브모듈에서 커밋
git add .
git commit -m "서브모듈 수정사항"

# 4. 원격 저장소에 푸시
git push origin main

# 5. 메인 프로젝트로 돌아가서 서브모듈 참조 업데이트
cd ../..
git add external/KLP-SignGlove
git commit -m "Update KLP-SignGlove submodule reference"
git push origin main
```

## 🚨 문제 해결

### 1. 서브모듈이 "detached HEAD" 상태인 경우

```bash
# 서브모듈로 이동
cd external/KLP-SignGlove

# main 브랜치로 체크아웃
git checkout main

# 최신 상태로 업데이트
git pull origin main
```

### 2. 서브모듈 업데이트가 안 되는 경우

```bash
# 서브모듈 초기화
git submodule deinit external/KLP-SignGlove
git submodule init external/KLP-SignGlove
git submodule update external/KLP-SignGlove
```

### 3. 서브모듈 충돌 해결

```bash
# 서브모듈의 충돌 해결
cd external/KLP-SignGlove
git status
git add .
git commit -m "Resolve conflicts"
```

## 📚 참고 자료

- [Git Submodule 공식 문서](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [Git Submodule 완벽 가이드](https://www.atlassian.com/git/tutorials/git-submodule)
- [서브모듈 모범 사례](https://github.com/blog/2104-working-with-submodules)

## 🔧 자동화 스크립트

### 업데이트 스크립트 (`update_submodules.sh`)

```bash
#!/bin/bash
echo "서브모듈 업데이트 시작..."

# 모든 서브모듈 업데이트
git submodule update --remote --merge

# 변경사항이 있으면 커밋
if ! git diff --quiet; then
    git add .
    git commit -m "Update submodules $(date)"
    git push origin main
    echo "서브모듈 업데이트 완료 및 푸시됨"
else
    echo "업데이트할 서브모듈이 없습니다"
fi
```

### 실행 권한 부여
```bash
chmod +x update_submodules.sh
```

---

**⚠️ 중요**: 이 문서는 SignGlove 프로젝트의 서브모듈 관리 가이드입니다. 서브모듈 작업 전에 반드시 이 가이드를 숙지하고 따라주세요.
