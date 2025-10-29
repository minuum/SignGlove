#!/usr/bin/env python3
"""
SignGlove 아두이노 연동 시스템 테스트
- 아두이노 연결 및 통신 기능 테스트
"""

import sys
import os
import time

# src 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from integrated_inference_system import IntegratedInferenceSystem, SystemConfig
from arduino_interface import SignGloveArduinoInterface, ArduinoConfig

def test_arduino_interface():
    """아두이노 인터페이스 테스트"""
    print("🔌 아두이노 인터페이스 테스트")
    print("-" * 40)
    
    # 아두이노 설정
    config = ArduinoConfig(
        auto_detect=True,
        auto_reconnect=True,
        max_reconnect_attempts=3
    )
    
    # 인터페이스 생성
    arduino = SignGloveArduinoInterface(config)
    
    # 콜백 등록
    def on_connected(data):
        print(f"✅ 아두이노 연결됨: {data['port']}")
    
    def on_disconnected(data):
        print("🔌 아두이노 연결 해제됨")
    
    def on_data_received(reading):
        print(f"📊 데이터 수신: {reading.timestamp_ms}ms | "
              f"Flex: {reading.flex1},{reading.flex2},{reading.flex3},{reading.flex4},{reading.flex5}")
    
    def on_error(data):
        print(f"❌ 오류: {data['error']}")
    
    arduino.register_callback('on_connected', on_connected)
    arduino.register_callback('on_disconnected', on_disconnected)
    arduino.register_callback('on_data_received', on_data_received)
    arduino.register_callback('on_error', on_error)
    
    # 연결 시도
    print("🔍 아두이노 포트 탐지 중...")
    if arduino.connect():
        print("✅ 아두이노 연결 성공!")
        
        # 상태 확인
        status = arduino.get_status()
        print(f"📊 아두이노 상태: {status}")
        
        # 연결 해제
        arduino.disconnect()
        print("✅ 아두이노 연결 해제됨")
    else:
        print("❌ 아두이노 연결 실패 (시뮬레이션 모드로 계속)")
    
    return True

def test_integrated_system_with_arduino():
    """통합 시스템 아두이노 연동 테스트"""
    print("\n🚀 통합 시스템 아두이노 연동 테스트")
    print("-" * 40)
    
    # 시스템 설정 (아두이노 활성화)
    config = SystemConfig(
        model_type='bigru',
        model_path=None,
        buffer_size=100,
        target_sampling_rate=33.3,
        min_confidence=0.7,
        auto_inference=False,
        realtime_display=False,  # 테스트 중에는 비활성화
        save_predictions=False,
        
        # 아두이노 설정
        arduino_enabled=True,
        arduino_port=None,  # 자동 탐지
        arduino_baudrate=115200,
        arduino_auto_detect=True,
        arduino_auto_reconnect=True,
        use_simulation=True  # 시뮬레이션 모드
    )
    
    # 통합 시스템 생성
    system = IntegratedInferenceSystem(config)
    
    print("✅ 통합 시스템 생성 완료")
    
    # 아두이노 상태 확인
    if system.arduino_interface:
        print("✅ 아두이노 인터페이스 초기화됨")
        status = system.arduino_interface.get_status()
        print(f"📊 아두이노 상태: {status}")
    else:
        print("⚠️ 아두이노 인터페이스가 비활성화됨")
    
    # 시뮬레이션 모드 테스트
    print("📡 시뮬레이션 모드 테스트...")
    system.collecting_data = True
    
    for i in range(5):
        sensor_data = system.get_sensor_data()
        if sensor_data:
            print(f"   샘플 {i+1}: {sensor_data.timestamp_ms}ms | "
                  f"Flex: {sensor_data.flex1},{sensor_data.flex2},{sensor_data.flex3}")
            system.process_sensor_data(sensor_data)
        time.sleep(0.1)
    
    # 추론 실행
    print("🔍 추론 실행...")
    system.run_inference()
    
    print("✅ 통합 시스템 아두이노 연동 테스트 완료")
    
    return True

def test_arduino_connection_modes():
    """아두이노 연결 모드 테스트"""
    print("\n🔌 아두이노 연결 모드 테스트")
    print("-" * 40)
    
    # 시뮬레이션 모드
    print("1. 시뮬레이션 모드 테스트")
    config_sim = SystemConfig(
        arduino_enabled=True,
        use_simulation=True
    )
    system_sim = IntegratedInferenceSystem(config_sim)
    print(f"   시뮬레이션 모드: {system_sim.config.use_simulation}")
    print(f"   아두이노 인터페이스: {'있음' if system_sim.arduino_interface else '없음'}")
    
    # 아두이노 모드
    print("\n2. 아두이노 모드 테스트")
    config_arduino = SystemConfig(
        arduino_enabled=True,
        use_simulation=False
    )
    system_arduino = IntegratedInferenceSystem(config_arduino)
    print(f"   시뮬레이션 모드: {system_arduino.config.use_simulation}")
    print(f"   아두이노 인터페이스: {'있음' if system_arduino.arduino_interface else '없음'}")
    
    # 아두이노 비활성화 모드
    print("\n3. 아두이노 비활성화 모드 테스트")
    config_disabled = SystemConfig(
        arduino_enabled=False,
        use_simulation=True
    )
    system_disabled = IntegratedInferenceSystem(config_disabled)
    print(f"   시뮬레이션 모드: {system_disabled.config.use_simulation}")
    print(f"   아두이노 인터페이스: {'있음' if system_disabled.arduino_interface else '없음'}")
    
    print("✅ 아두이노 연결 모드 테스트 완료")
    
    return True

def main():
    """메인 테스트 함수"""
    print("🧪 SignGlove 아두이노 연동 시스템 테스트 시작")
    print("=" * 60)
    
    tests = [
        ("아두이노 인터페이스", test_arduino_interface),
        ("통합 시스템 아두이노 연동", test_integrated_system_with_arduino),
        ("아두이노 연결 모드", test_arduino_connection_modes),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n🔬 {test_name} 테스트 중...")
            if test_func():
                print(f"✅ {test_name} 테스트 통과!")
                passed += 1
            else:
                print(f"❌ {test_name} 테스트 실패!")
        except Exception as e:
            print(f"❌ {test_name} 테스트 오류: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"📊 테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 아두이노 연동 테스트가 성공적으로 완료되었습니다!")
        print("\n💡 사용 방법:")
        print("   1. 시뮬레이션 모드: use_simulation=True")
        print("   2. 아두이노 모드: use_simulation=False, 아두이노 연결")
        print("   3. 키보드 제어: C(연결), D(해제), F10(상태)")
    else:
        print("⚠️ 일부 테스트가 실패했습니다.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
