#!/usr/bin/env python3
"""
SignGlove 전체 데모 실행 스크립트
서버 시작부터 테스트까지 자동화
"""

import os
import sys
import asyncio
import subprocess
import time
import threading
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tests.scenarios.test_scenario_runner import TestScenarioRunner

class ServerManager:
    """서버 관리 클래스"""
    
    def __init__(self):
        self.server_process = None
        self.server_ready = False
    
    def start_server(self):
        """서버 시작"""
        print("🚀 SignGlove 서버 시작 중...")
        
        # 데이터 디렉토리 생성
        data_dirs = ["data/raw", "data/processed", "data/backup"]
        for data_dir in data_dirs:
            Path(data_dir).mkdir(parents=True, exist_ok=True)
        
        # 서버 실행
        server_cmd = [
            sys.executable, "-m", "uvicorn", 
            "server.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ]
        
        self.server_process = subprocess.Popen(
            server_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            cwd=project_root
        )
        
        # 서버 시작 대기
        self._wait_for_server()
    
    def _wait_for_server(self):
        """서버 시작 대기"""
        import httpx
        
        for i in range(30):  # 30초 대기
            try:
                response = httpx.get("http://localhost:8000/health", timeout=2.0)
                if response.status_code == 200:
                    self.server_ready = True
                    print("✅ 서버 시작 완료")
                    return
            except:
                pass
            
            time.sleep(1)
            print(f"서버 시작 대기 중... ({i+1}/30)")
        
        print("❌ 서버 시작 실패")
        self.stop_server()
    
    def stop_server(self):
        """서버 중지"""
        if self.server_process:
            print("🛑 서버 중지 중...")
            self.server_process.terminate()
            self.server_process.wait()
            print("✅ 서버 중지 완료")
    
    def is_ready(self):
        """서버 준비 상태 확인"""
        return self.server_ready

async def run_demo_scenarios():
    """데모 시나리오 실행"""
    print("\n🎯 데모 시나리오 시작")
    
    runner = TestScenarioRunner()
    
    # 단계별 데모 실행
    scenarios = [
        ("연결성 확인", runner.scenario_1_basic_connectivity),
        ("센서 데이터 테스트", runner.scenario_2_sensor_data_test),
        ("제스처 데이터 테스트", runner.scenario_3_gesture_data_test),
        ("성능 테스트", runner.scenario_4_performance_test),
        ("데이터 저장 확인", runner.scenario_6_storage_test)
    ]
    
    for name, scenario_func in scenarios:
        print(f"\n▶️ {name} 시작")
        try:
            result = await scenario_func()
            if result:
                print(f"✅ {name} 성공")
            else:
                print(f"❌ {name} 실패")
        except Exception as e:
            print(f"❌ {name} 오류: {str(e)}")
        
        # 다음 시나리오 전 잠시 대기
        await asyncio.sleep(1)

async def run_data_generation_demo():
    """데이터 생성 데모"""
    print("\n📊 데이터 생성 데모 시작")
    
    from tests.dummy_data_generator import DummyDataGenerator
    
    generator = DummyDataGenerator()
    
    # 다양한 데이터 생성
    print("🔸 센서 데이터 생성 및 전송 (20개)...")
    await generator.run_sensor_simulation(duration=10, sample_rate=2)
    
    print("\n🔸 제스처 데이터 생성 및 전송 (한국어 모음/자음/숫자)...")
    await generator.run_gesture_simulation(gesture_count=15)
    
    print("\n📈 최종 데이터 통계:")
    stats = await TestScenarioRunner().get_data_stats()
    print(f"  - 센서 데이터: {stats.get('total_sensor_records', 0)}개")
    print(f"  - 제스처 데이터: {stats.get('total_gesture_records', 0)}개")
    print(f"  - 디스크 사용량: {stats.get('disk_usage', {}).get('total_mb', 0):.2f} MB")

def print_demo_info():
    """데모 정보 출력"""
    print("="*60)
    print("           🤖 SignGlove 시스템 데모")
    print("="*60)
    print("이 데모는 다음과 같은 기능을 테스트합니다:")
    print("  1. FastAPI 서버 자동 시작")
    print("  2. 센서 데이터 수집 및 검증")
    print("  3. 제스처 데이터 처리")
    print("  4. CSV 파일 저장")
    print("  5. 성능 테스트")
    print("  6. 더미 데이터 생성")
    print("="*60)

async def main():
    """메인 함수"""
    print_demo_info()
    
    # 프로젝트 루트로 이동
    os.chdir(project_root)
    
    # 서버 매니저 생성
    server_manager = ServerManager()
    
    try:
        # 서버 시작
        server_manager.start_server()
        
        if not server_manager.is_ready():
            print("❌ 서버 시작 실패 - 데모 중단")
            return 1
        
        print("\n🎉 서버 준비 완료! 데모 시작")
        
        # 데모 옵션 선택
        print("\n데모 옵션을 선택하세요:")
        print("1. 전체 데모 (추천)")
        print("2. 시나리오 테스트만")
        print("3. 데이터 생성 데모만")
        print("4. 대화형 모드")
        
        choice = input("\n선택하세요 (1-4): ")
        
        if choice == "1":
            # 전체 데모
            await run_demo_scenarios()
            await run_data_generation_demo()
        elif choice == "2":
            # 시나리오 테스트만
            await run_demo_scenarios()
        elif choice == "3":
            # 데이터 생성 데모만
            await run_data_generation_demo()
        elif choice == "4":
            # 대화형 모드
            print("\n🎮 대화형 모드 - 서버가 실행 중입니다")
            print("다른 터미널에서 다음을 실행하세요:")
            print("  python scripts/run_tests.py")
            print("또는")
            print("  python tests/dummy_data_generator.py")
            print("\n서버 종료: Ctrl+C")
            
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n사용자가 종료했습니다.")
        else:
            print("❌ 잘못된 선택입니다.")
            return 1
        
        print("\n🎉 데모 완료!")
        
    except KeyboardInterrupt:
        print("\n사용자가 데모를 중단했습니다.")
    except Exception as e:
        print(f"❌ 데모 실행 중 오류: {str(e)}")
        return 1
    finally:
        # 서버 중지
        server_manager.stop_server()
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 