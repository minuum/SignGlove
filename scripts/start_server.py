#!/usr/bin/env python3
"""
SignGlove 서버 시작 스크립트
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """서버 시작 메인 함수"""
    print("🚀 SignGlove 서버 시작")
    
    # 프로젝트 루트 디렉토리로 이동
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # 데이터 디렉토리 생성
    data_dirs = ["data/raw", "data/processed", "data/backup"]
    for data_dir in data_dirs:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # 서버 실행
    server_cmd = [
        sys.executable, "-m", "uvicorn", 
        "server.main:app", 
        "--reload", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ]
    
    print(f"서버 명령어: {' '.join(server_cmd)}")
    print("서버 종료: Ctrl+C")
    print("=" * 50)
    
    try:
        subprocess.run(server_cmd, check=True)
    except KeyboardInterrupt:
        print("\n서버 종료됨")
    except subprocess.CalledProcessError as e:
        print(f"서버 실행 오류: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 