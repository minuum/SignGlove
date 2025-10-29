#!/bin/bash

# SignGlove 프로젝트 서브모듈 자동 업데이트 스크립트
# 작성자: Billy
# 날짜: 2025-10-29

set -e  # 오류 발생 시 스크립트 중단

echo "🚀 SignGlove 서브모듈 업데이트 시작..."
echo "=================================="

# 현재 디렉토리가 SignGlove 프로젝트 루트인지 확인
if [ ! -f "README.md" ] || [ ! -d "external" ]; then
    echo "❌ 오류: SignGlove 프로젝트 루트 디렉토리에서 실행해주세요"
    echo "현재 디렉토리: $(pwd)"
    exit 1
fi

# Git 상태 확인
if ! git status > /dev/null 2>&1; then
    echo "❌ 오류: Git 저장소가 아닙니다"
    exit 1
fi

echo "📁 현재 디렉토리: $(pwd)"
echo ""

# 서브모듈 상태 확인
echo "🔍 서브모듈 상태 확인 중..."
if [ -f ".gitmodules" ]; then
    echo "✅ .gitmodules 파일 발견"
    git submodule status
else
    echo "⚠️  .gitmodules 파일이 없습니다. 서브모듈이 설정되지 않았을 수 있습니다."
fi
echo ""

# 서브모듈 업데이트
echo "🔄 서브모듈 업데이트 중..."
if [ -f ".gitmodules" ]; then
    # 서브모듈이 설정된 경우
    git submodule update --remote --merge
    
    # 변경사항 확인
    if ! git diff --quiet; then
        echo "📝 서브모듈 변경사항 발견"
        git add .
        git commit -m "Update submodules $(date '+%Y-%m-%d %H:%M:%S')"
        
        # 원격 저장소에 푸시할지 확인
        read -p "원격 저장소에 푸시하시겠습니까? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git push origin main
            echo "✅ 서브모듈 업데이트 완료 및 푸시됨"
        else
            echo "✅ 서브모듈 업데이트 완료 (로컬에만 커밋됨)"
        fi
    else
        echo "✅ 업데이트할 서브모듈이 없습니다"
    fi
else
    echo "⚠️  서브모듈이 설정되지 않았습니다"
    echo "다음 명령어로 서브모듈을 설정하세요:"
    echo "  git submodule add <repository-url> external/<folder-name>"
fi

echo ""
echo "🎉 서브모듈 업데이트 작업 완료!"
echo "=================================="
