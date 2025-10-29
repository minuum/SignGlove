#!/bin/bash

# SignGlove 프로젝트 서브모듈 설정 스크립트
# 작성자: Billy
# 날짜: 2025-10-29

set -e  # 오류 발생 시 스크립트 중단

echo "🔧 SignGlove 서브모듈 설정 시작..."
echo "=================================="

# 현재 디렉토리가 SignGlove 프로젝트 루트인지 확인
if [ ! -f "README.md" ] || [ ! -d "external" ]; then
    echo "❌ 오류: SignGlove 프로젝트 루트 디렉토리에서 실행해주세요"
    echo "현재 디렉토리: $(pwd)"
    exit 1
fi

echo "📁 현재 디렉토리: $(pwd)"
echo ""

# 기존 external 폴더 백업
echo "💾 기존 external 폴더 백업 중..."
if [ -d "external_backup" ]; then
    echo "⚠️  external_backup 폴더가 이미 존재합니다. 삭제하시겠습니까? (y/N)"
    read -p "삭제하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf external_backup
    else
        echo "❌ 백업 폴더가 존재하여 중단합니다"
        exit 1
    fi
fi

cp -r external external_backup
echo "✅ 백업 완료: external_backup/"
echo ""

# 서브모듈 설정
echo "🔗 서브모듈 설정 중..."

# KLP-SignGlove 서브모듈 설정
if [ -d "external/KLP-SignGlove" ]; then
    echo "📦 KLP-SignGlove 서브모듈 설정 중..."
    # 기존 폴더를 임시로 이동
    mv external/KLP-SignGlove external/KLP-SignGlove_temp
    
    # 서브모듈로 추가
    git submodule add https://github.com/Kyle-Riss/KLP-SignGlove.git external/KLP-SignGlove
    
    # 기존 변경사항이 있다면 복사
    if [ -d "external/KLP-SignGlove_temp" ]; then
        echo "📋 기존 변경사항 복사 중..."
        cp -r external/KLP-SignGlove_temp/* external/KLP-SignGlove/ 2>/dev/null || true
        cp -r external/KLP-SignGlove_temp/.* external/KLP-SignGlove/ 2>/dev/null || true
        rm -rf external/KLP-SignGlove_temp
    fi
    
    echo "✅ KLP-SignGlove 서브모듈 설정 완료"
fi

# SignGlove_HW 서브모듈 설정
if [ -d "external/SignGlove_HW" ]; then
    echo "📦 SignGlove_HW 서브모듈 설정 중..."
    # 실제 저장소 URL을 확인하고 설정
    echo "⚠️  SignGlove_HW의 실제 GitHub 저장소 URL을 입력해주세요:"
    read -p "저장소 URL: " hw_repo_url
    
    if [ -n "$hw_repo_url" ]; then
        # 기존 폴더를 임시로 이동
        mv external/SignGlove_HW external/SignGlove_HW_temp
        
        # 서브모듈로 추가
        git submodule add "$hw_repo_url" external/SignGlove_HW
        
        # 기존 변경사항이 있다면 복사
        if [ -d "external/SignGlove_HW_temp" ]; then
            echo "📋 기존 변경사항 복사 중..."
            cp -r external/SignGlove_HW_temp/* external/SignGlove_HW/ 2>/dev/null || true
            cp -r external/SignGlove_HW_temp/.* external/SignGlove_HW/ 2>/dev/null || true
            rm -rf external/SignGlove_HW_temp
        fi
        
        echo "✅ SignGlove_HW 서브모듈 설정 완료"
    else
        echo "⚠️  SignGlove_HW 서브모듈 설정 건너뜀"
    fi
fi

# SignGlove-DataAnalysis 서브모듈 설정
if [ -d "external/SignGlove-DataAnalysis" ]; then
    echo "📦 SignGlove-DataAnalysis 서브모듈 설정 중..."
    echo "⚠️  SignGlove-DataAnalysis의 실제 GitHub 저장소 URL을 입력해주세요:"
    read -p "저장소 URL: " analysis_repo_url
    
    if [ -n "$analysis_repo_url" ]; then
        # 기존 폴더를 임시로 이동
        mv external/SignGlove-DataAnalysis external/SignGlove-DataAnalysis_temp
        
        # 서브모듈로 추가
        git submodule add "$analysis_repo_url" external/SignGlove-DataAnalysis
        
        # 기존 변경사항이 있다면 복사
        if [ -d "external/SignGlove-DataAnalysis_temp" ]; then
            echo "📋 기존 변경사항 복사 중..."
            cp -r external/SignGlove-DataAnalysis_temp/* external/SignGlove-DataAnalysis/ 2>/dev/null || true
            cp -r external/SignGlove-DataAnalysis_temp/.* external/SignGlove-DataAnalysis/ 2>/dev/null || true
            rm -rf external/SignGlove-DataAnalysis_temp
        fi
        
        echo "✅ SignGlove-DataAnalysis 서브모듈 설정 완료"
    else
        echo "⚠️  SignGlove-DataAnalysis 서브모듈 설정 건너뜀"
    fi
fi

echo ""
echo "🔍 서브모듈 상태 확인..."
git submodule status

echo ""
echo "📝 .gitmodules 파일 내용:"
if [ -f ".gitmodules" ]; then
    cat .gitmodules
else
    echo "⚠️  .gitmodules 파일이 생성되지 않았습니다"
fi

echo ""
echo "🎉 서브모듈 설정 완료!"
echo "=================================="
echo ""
echo "다음 단계:"
echo "1. git add ."
echo "2. git commit -m 'Add submodules'"
echo "3. git push origin main"
echo ""
echo "서브모듈 업데이트는 다음 명령어를 사용하세요:"
echo "./external/update_submodules.sh"
