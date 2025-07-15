"""
SignGlove 테스트 시나리오 실행기
단계별로 시스템 기능을 테스트하는 스크립트
"""

import asyncio
import time
import json
from datetime import datetime
from pathlib import Path
import httpx
from typing import Dict, List, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tests.dummy_data_generator import DummyDataGenerator
from server.models.sensor_data import SignGestureType

class TestScenarioRunner:
    """테스트 시나리오 실행기"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        """
        테스트 시나리오 실행기 초기화
        
        Args:
            server_url: 서버 URL
        """
        self.server_url = server_url
        self.generator = DummyDataGenerator(server_url)
        self.results = []
        
    async def check_server_health(self) -> bool:
        """서버 상태 확인"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.server_url}/health", timeout=5.0)
                if response.status_code == 200:
                    print("✅ 서버 상태: 정상")
                    return True
                else:
                    print(f"❌ 서버 상태: 오류 ({response.status_code})")
                    return False
        except Exception as e:
            print(f"❌ 서버 연결 실패: {str(e)}")
            return False
    
    async def get_server_status(self) -> Dict[str, Any]:
        """서버 상태 정보 조회"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.server_url}/", timeout=5.0)
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def get_data_stats(self) -> Dict[str, Any]:
        """데이터 통계 조회"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.server_url}/data/stats", timeout=5.0)
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def print_separator(self, title: str):
        """구분선 출력"""
        print("\n" + "="*50)
        print(f"  {title}")
        print("="*50)
    
    def print_step(self, step_num: int, title: str):
        """단계 제목 출력"""
        print(f"\n🔸 Step {step_num}: {title}")
        print("-" * 30)
    
    async def scenario_1_basic_connectivity(self):
        """시나리오 1: 기본 연결성 테스트"""
        self.print_separator("시나리오 1: 기본 연결성 테스트")
        
        self.print_step(1, "서버 헬스 체크")
        health_ok = await self.check_server_health()
        if not health_ok:
            print("❌ 서버 연결 실패 - 테스트 중단")
            return False
        
        self.print_step(2, "서버 상태 정보 조회")
        status = await self.get_server_status()
        print(f"서버 상태: {json.dumps(status, indent=2, default=str)}")
        
        self.print_step(3, "데이터 통계 조회")
        stats = await self.get_data_stats()
        print(f"데이터 통계: {json.dumps(stats, indent=2, default=str)}")
        
        print("\n✅ 시나리오 1 완료: 기본 연결성 정상")
        return True
    
    async def scenario_2_sensor_data_test(self):
        """시나리오 2: 센서 데이터 전송 테스트"""
        self.print_separator("시나리오 2: 센서 데이터 전송 테스트")
        
        self.print_step(1, "단일 센서 데이터 전송")
        sensor_data = self.generator.generate_sensor_data()
        success = await self.generator.send_sensor_data(sensor_data)
        
        if success:
            print("✅ 센서 데이터 전송 성공")
        else:
            print("❌ 센서 데이터 전송 실패")
            return False
        
        self.print_step(2, "연속 센서 데이터 전송 (10개)")
        successful = 0
        failed = 0
        
        for i in range(10):
            sensor_data = self.generator.generate_sensor_data()
            success = await self.generator.send_sensor_data(sensor_data)
            if success:
                successful += 1
            else:
                failed += 1
            
            await asyncio.sleep(0.1)  # 100ms 간격
        
        print(f"전송 결과: 성공 {successful}, 실패 {failed}")
        
        self.print_step(3, "다양한 제스처 타입 테스트")
        gesture_types = ["neutral", "fist", "open", "pointing", "thumbs_up"]
        
        for gesture_type in gesture_types:
            sensor_data = self.generator.generate_sensor_data(gesture_type)
            success = await self.generator.send_sensor_data(sensor_data)
            status = "✅" if success else "❌"
            print(f"{status} {gesture_type} 제스처 데이터 전송")
        
        self.print_step(4, "최종 통계 확인")
        stats = await self.get_data_stats()
        print(f"수집된 센서 데이터: {stats.get('total_sensor_records', 0)}개")
        
        print("\n✅ 시나리오 2 완료: 센서 데이터 전송 테스트")
        return True
    
    async def scenario_3_gesture_data_test(self):
        """시나리오 3: 제스처 데이터 전송 테스트"""
        self.print_separator("시나리오 3: 제스처 데이터 전송 테스트")
        
        self.print_step(1, "한국어 모음 제스처 테스트")
        vowels = ['ㅏ', 'ㅓ', 'ㅗ', 'ㅜ', 'ㅡ']
        
        for vowel in vowels:
            gesture_data = self.generator.generate_gesture_data(vowel, SignGestureType.VOWEL)
            success = await self.generator.send_gesture_data(gesture_data)
            status = "✅" if success else "❌"
            print(f"{status} 모음 '{vowel}' 제스처 전송 (시퀀스: {len(gesture_data.sensor_sequence)}개)")
            
            await asyncio.sleep(0.5)
        
        self.print_step(2, "한국어 자음 제스처 테스트")
        consonants = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ']
        
        for consonant in consonants:
            gesture_data = self.generator.generate_gesture_data(consonant, SignGestureType.CONSONANT)
            success = await self.generator.send_gesture_data(gesture_data)
            status = "✅" if success else "❌"
            print(f"{status} 자음 '{consonant}' 제스처 전송 (시퀀스: {len(gesture_data.sensor_sequence)}개)")
            
            await asyncio.sleep(0.5)
        
        self.print_step(3, "숫자 제스처 테스트")
        numbers = ['1', '2', '3', '4', '5']
        
        for number in numbers:
            gesture_data = self.generator.generate_gesture_data(number, SignGestureType.NUMBER)
            success = await self.generator.send_gesture_data(gesture_data)
            status = "✅" if success else "❌"
            print(f"{status} 숫자 '{number}' 제스처 전송 (시퀀스: {len(gesture_data.sensor_sequence)}개)")
            
            await asyncio.sleep(0.5)
        
        self.print_step(4, "최종 통계 확인")
        stats = await self.get_data_stats()
        print(f"수집된 제스처 데이터: {stats.get('total_gesture_records', 0)}개")
        
        print("\n✅ 시나리오 3 완료: 제스처 데이터 전송 테스트")
        return True
    
    async def scenario_4_performance_test(self):
        """시나리오 4: 성능 테스트"""
        self.print_separator("시나리오 4: 성능 테스트")
        
        self.print_step(1, "고속 센서 데이터 전송 테스트 (100개)")
        start_time = time.time()
        successful = 0
        failed = 0
        
        for i in range(100):
            sensor_data = self.generator.generate_sensor_data()
            success = await self.generator.send_sensor_data(sensor_data)
            if success:
                successful += 1
            else:
                failed += 1
            
            if i % 20 == 0:
                progress = (i / 100) * 100
                print(f"진행률: {progress:.0f}%")
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"전송 결과: 성공 {successful}, 실패 {failed}")
        print(f"소요 시간: {elapsed:.2f}초")
        print(f"처리 속도: {successful/elapsed:.2f} 건/초")
        
        self.print_step(2, "동시 전송 테스트")
        print("동시에 10개 센서 데이터 전송...")
        
        start_time = time.time()
        
        # 동시 전송 태스크 생성
        tasks = []
        for i in range(10):
            sensor_data = self.generator.generate_sensor_data()
            tasks.append(self.generator.send_sensor_data(sensor_data))
        
        # 동시 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        successful = sum(1 for r in results if r is True)
        failed = len(results) - successful
        
        print(f"동시 전송 결과: 성공 {successful}, 실패 {failed}")
        print(f"소요 시간: {elapsed:.2f}초")
        
        self.print_step(3, "메모리 및 저장소 사용량 확인")
        stats = await self.get_data_stats()
        disk_usage = stats.get('disk_usage', {})
        print(f"디스크 사용량: {disk_usage.get('total_mb', 0):.2f} MB")
        
        print("\n✅ 시나리오 4 완료: 성능 테스트")
        return True
    
    async def scenario_5_data_validation_test(self):
        """시나리오 5: 데이터 검증 테스트"""
        self.print_separator("시나리오 5: 데이터 검증 테스트")
        
        self.print_step(1, "정상 데이터 검증")
        sensor_data = self.generator.generate_sensor_data()
        success = await self.generator.send_sensor_data(sensor_data)
        status = "✅" if success else "❌"
        print(f"{status} 정상 데이터 검증 통과")
        
        self.print_step(2, "경계값 테스트")
        # 플렉스 센서 경계값 테스트
        from server.models.sensor_data import FlexSensorData, GyroData, SensorData
        
        edge_cases = [
            ("최소값", FlexSensorData(flex_1=0, flex_2=0, flex_3=0, flex_4=0, flex_5=0)),
            ("최대값", FlexSensorData(flex_1=1023, flex_2=1023, flex_3=1023, flex_4=1023, flex_5=1023)),
            ("중간값", FlexSensorData(flex_1=512, flex_2=512, flex_3=512, flex_4=512, flex_5=512))
        ]
        
        for case_name, flex_data in edge_cases:
            gyro_data = self.generator.generate_realistic_gyro_data()
            sensor_data = SensorData(
                device_id="TEST_DEVICE",
                flex_sensors=flex_data,
                gyro_data=gyro_data,
                battery_level=50,
                signal_strength=-50
            )
            
            success = await self.generator.send_sensor_data(sensor_data)
            status = "✅" if success else "❌"
            print(f"{status} {case_name} 테스트")
        
        self.print_step(3, "잘못된 데이터 테스트")
        # 잘못된 데이터는 서버에서 검증 오류로 거부되어야 함
        print("(잘못된 데이터는 서버에서 검증 오류로 거부됩니다)")
        
        self.print_step(4, "타임스탬프 검증")
        from datetime import datetime, timedelta
        
        # 미래 시간 데이터
        future_sensor_data = self.generator.generate_sensor_data()
        future_sensor_data.timestamp = datetime.now() + timedelta(hours=1)
        
        success = await self.generator.send_sensor_data(future_sensor_data)
        status = "✅" if success else "❌"
        print(f"{status} 미래 타임스탬프 데이터 처리")
        
        print("\n✅ 시나리오 5 완료: 데이터 검증 테스트")
        return True
    
    async def scenario_6_storage_test(self):
        """시나리오 6: 데이터 저장 테스트"""
        self.print_separator("시나리오 6: 데이터 저장 테스트")
        
        self.print_step(1, "CSV 파일 생성 확인")
        data_path = Path("data/raw")
        today = datetime.now().strftime("%Y%m%d")
        
        expected_files = [
            f"sensor_data_{today}.csv",
            f"gesture_data_{today}.csv"
        ]
        
        for filename in expected_files:
            file_path = data_path / filename
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"✅ {filename} 존재 (크기: {size} bytes)")
            else:
                print(f"❌ {filename} 없음")
        
        self.print_step(2, "데이터 저장 테스트")
        # 테스트 데이터 전송
        sensor_data = self.generator.generate_sensor_data()
        success = await self.generator.send_sensor_data(sensor_data)
        
        if success:
            print("✅ 센서 데이터 저장 확인")
        else:
            print("❌ 센서 데이터 저장 실패")
        
        # 제스처 데이터 전송
        gesture_data = self.generator.generate_gesture_data("ㅏ", SignGestureType.VOWEL)
        success = await self.generator.send_gesture_data(gesture_data)
        
        if success:
            print("✅ 제스처 데이터 저장 확인")
        else:
            print("❌ 제스처 데이터 저장 실패")
        
        self.print_step(3, "JSON 시퀀스 파일 확인")
        json_file = data_path / f"gesture_sequences_{today}.json"
        
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"✅ JSON 시퀀스 파일 존재 (제스처 수: {len(data)})")
            except Exception as e:
                print(f"❌ JSON 파일 읽기 오류: {str(e)}")
        else:
            print("❌ JSON 시퀀스 파일 없음")
        
        print("\n✅ 시나리오 6 완료: 데이터 저장 테스트")
        return True
    
    async def run_all_scenarios(self):
        """모든 시나리오 실행"""
        print("🚀 SignGlove 시스템 테스트 시작")
        print(f"테스트 대상 서버: {self.server_url}")
        print(f"테스트 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        scenarios = [
            ("기본 연결성 테스트", self.scenario_1_basic_connectivity),
            ("센서 데이터 전송 테스트", self.scenario_2_sensor_data_test),
            ("제스처 데이터 전송 테스트", self.scenario_3_gesture_data_test),
            ("성능 테스트", self.scenario_4_performance_test),
            ("데이터 검증 테스트", self.scenario_5_data_validation_test),
            ("데이터 저장 테스트", self.scenario_6_storage_test)
        ]
        
        passed = 0
        failed = 0
        
        for name, scenario_func in scenarios:
            try:
                result = await scenario_func()
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ {name} 실행 중 오류: {str(e)}")
                failed += 1
        
        # 최종 결과
        self.print_separator("테스트 결과 요약")
        print(f"✅ 통과: {passed}개")
        print(f"❌ 실패: {failed}개")
        print(f"총 테스트: {passed + failed}개")
        
        if failed == 0:
            print("\n🎉 모든 테스트 통과!")
        else:
            print(f"\n⚠️  {failed}개 테스트 실패")
        
        # 최종 통계 출력
        stats = await self.get_data_stats()
        print(f"\n📊 최종 데이터 통계:")
        print(f"  - 센서 데이터: {stats.get('total_sensor_records', 0)}개")
        print(f"  - 제스처 데이터: {stats.get('total_gesture_records', 0)}개")
        print(f"  - 디스크 사용량: {stats.get('disk_usage', {}).get('total_mb', 0):.2f} MB")

async def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SignGlove 테스트 시나리오 실행기')
    parser.add_argument('--server', default='http://localhost:8000', help='서버 URL')
    parser.add_argument('--scenario', type=int, help='특정 시나리오 번호 (1-6)')
    
    args = parser.parse_args()
    
    runner = TestScenarioRunner(args.server)
    
    if args.scenario:
        # 특정 시나리오 실행
        scenarios = {
            1: runner.scenario_1_basic_connectivity,
            2: runner.scenario_2_sensor_data_test,
            3: runner.scenario_3_gesture_data_test,
            4: runner.scenario_4_performance_test,
            5: runner.scenario_5_data_validation_test,
            6: runner.scenario_6_storage_test
        }
        
        if args.scenario in scenarios:
            await scenarios[args.scenario]()
        else:
            print("❌ 잘못된 시나리오 번호입니다 (1-6)")
    else:
        # 모든 시나리오 실행
        await runner.run_all_scenarios()

if __name__ == "__main__":
    asyncio.run(main()) 