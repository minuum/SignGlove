#!/usr/bin/env python3
"""
SignGlove 테스트 실행 스크립트
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tests.scenarios.test_scenario_runner import TestScenarioRunner

def check_server_running():
    """서버가 실행 중인지 확인"""
    try:
        import httpx
        response = httpx.get("http://localhost:8000/health", timeout=2.0)
        return response.status_code == 200
    except:
        return False

async def main():
    """테스트 실행 메인 함수"""
    print("🧪 SignGlove 테스트 실행")
    
    # 프로젝트 루트로 이동
    os.chdir(project_root)
    
    # 서버 실행 확인
    if not check_server_running():
        print("❌ 서버가 실행되지 않았습니다.")
        print("먼저 서버를 실행해주세요: python scripts/start_server.py")
        return 1
    
    print("✅ 서버가 실행 중입니다.")
    
    # 테스트 옵션 선택
    print("\n테스트 옵션을 선택하세요:")
    print("1. 전체 테스트 실행")
    print("2. 기본 연결성 테스트")
    print("3. 센서 데이터 테스트")
    print("4. 제스처 데이터 테스트")
    print("5. 성능 테스트")
    print("6. 데이터 검증 테스트")
    print("7. 데이터 저장 테스트")
    print("8. 더미 데이터 생성기 실행")
    
    choice = input("\n선택하세요 (1-8): ")
    
    runner = TestScenarioRunner()
    
    try:
        if choice == "1":
            await runner.run_all_scenarios()
        elif choice == "2":
            await runner.scenario_1_basic_connectivity()
        elif choice == "3":
            await runner.scenario_2_sensor_data_test()
        elif choice == "4":
            await runner.scenario_3_gesture_data_test()
        elif choice == "5":
            await runner.scenario_4_performance_test()
        elif choice == "6":
            await runner.scenario_5_data_validation_test()
        elif choice == "7":
            await runner.scenario_6_storage_test()
        elif choice == "8":
            # 더미 데이터 생성기 실행
            from tests.dummy_data_generator import main as dummy_main
            dummy_main()
        else:
            print("❌ 잘못된 선택입니다.")
            return 1
            
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {str(e)}")
        return 1
    
    print("\n테스트 완료!")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 