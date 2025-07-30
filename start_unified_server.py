#!/usr/bin/env python3
"""
SignGlove 통합 서버 실행 스크립트
이민우님이 사용할 서버 시작 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent))

from server.unified_server import start_server

if __name__ == "__main__":
    print("🚀 SignGlove 통합 데이터 수집 서버를 시작합니다...")
    
    # 서버 시작 (기본: localhost:8000)
    start_server(host="0.0.0.0", port=8000) 