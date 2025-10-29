#!/usr/bin/env python3
"""
SignGlove 추론 시스템 테스트
- 통합 추론 시스템의 기본 기능 테스트
"""

import sys
import os
import time
import numpy as np

# src 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from integrated_inference_system import IntegratedInferenceSystem, SystemConfig
from inference_engine import SignGloveInferenceEngine, create_model_config
from data_buffer import SignGloveDataBuffer, SensorReading

def test_inference_engine():
    """추론 엔진 테스트"""
    print("🤖 추론 엔진 테스트")
    print("-" * 40)
    
    # 모델 설정
    config = create_model_config('bigru')
    
    # 추론 엔진 생성
    engine = SignGloveInferenceEngine(config)
    
    # 테스트 데이터 생성
    test_data = np.random.randn(80, 8)  # 80 time steps, 8 features
    
    # 추론 실행
    result = engine.predict(test_data)
    
    print(f"✅ 예측 결과: {result.predicted_class}")
    print(f"✅ 신뢰도: {result.confidence:.3f}")
    print(f"✅ 처리 시간: {result.processing_time:.3f}초")
    print(f"✅ 모델 타입: {result.model_type}")
    
    # 모델 정보 출력
    model_info = engine.get_model_info()
    print(f"✅ 모델 파라미터 수: {model_info['total_parameters']:,}")
    print(f"✅ 클래스 수: {model_info['num_classes']}")
    
    return True

def test_data_buffer():
    """데이터 버퍼 테스트"""
    print("\n📊 데이터 버퍼 테스트")
    print("-" * 40)
    
    # 버퍼 생성
    buffer = SignGloveDataBuffer(max_size=100, target_sampling_rate=33.3)
    
    # 콜백 등록
    def on_warning(data):
        print(f"⚠️ 버퍼 경고: {data['usage']*100:.1f}% 사용")
    
    def on_full(data):
        print(f"🔴 버퍼 포화: {data['usage']*100:.1f}% 사용")
    
    buffer.register_callback('on_buffer_warning', on_warning)
    buffer.register_callback('on_buffer_full', on_full)
    
    # 모니터링 시작
    buffer.start_monitoring()
    
    # 테스트 데이터 생성 및 추가
    print("📡 테스트 데이터 생성 중...")
    for i in range(120):  # 버퍼 크기보다 많은 데이터
        reading = SensorReading(
            timestamp_ms=int(time.time() * 1000),
            recv_timestamp_ms=int(time.time() * 1000),
            flex1=np.random.randint(200, 800),
            flex2=np.random.randint(200, 800),
            flex3=np.random.randint(200, 800),
            flex4=np.random.randint(200, 800),
            flex5=np.random.randint(200, 800),
            pitch=np.random.uniform(-180, 180),
            roll=np.random.uniform(-90, 90),
            yaw=np.random.uniform(-180, 180),
            sampling_hz=33.3
        )
        
        buffer.add_data(reading)
        time.sleep(0.01)  # 빠른 테스트를 위해 0.01초
    
    # 통계 출력
    stats = buffer.get_stats()
    print(f"✅ 총 샘플: {stats['total_samples']}개")
    print(f"✅ 손실 샘플: {stats['dropped_samples']}개")
    print(f"✅ 버퍼 사용률: {stats['buffer_usage']*100:.1f}%")
    print(f"✅ 평균 샘플링 레이트: {stats['avg_sampling_rate']:.1f}Hz")
    
    # 시퀀스 데이터 가져오기 테스트
    sequence = buffer.get_latest_sequence(10)
    print(f"✅ 시퀀스 데이터: {len(sequence)}개 샘플")
    
    # 모니터링 중지
    buffer.stop_monitoring()
    
    return True

def test_integrated_system():
    """통합 시스템 테스트"""
    print("\n🚀 통합 추론 시스템 테스트")
    print("-" * 40)
    
    # 시스템 설정
    config = SystemConfig(
        model_type='bigru',
        model_path=None,  # Mock 모드
        buffer_size=100,
        target_sampling_rate=33.3,
        min_confidence=0.7,
        auto_inference=False,
        realtime_display=False,  # 테스트 중에는 비활성화
        save_predictions=False
    )
    
    # 통합 시스템 생성
    system = IntegratedInferenceSystem(config)
    
    print("✅ 통합 시스템 생성 완료")
    
    # 데이터 수집 시뮬레이션
    print("📡 데이터 수집 시뮬레이션...")
    system.collecting_data = True
    
    for i in range(50):
        sensor_data = system.simulate_sensor_data()
        if sensor_data:
            system.process_sensor_data(sensor_data)
        time.sleep(0.01)
    
    # 추론 실행
    print("🔍 추론 실행...")
    system.run_inference()
    
    # 통계 확인
    print("📊 통계 확인...")
    system._show_stats()
    
    # 조합 상태 확인
    print("📝 조합 상태 확인...")
    system._show_composition()
    
    print("✅ 통합 시스템 테스트 완료")
    
    return True

def test_korean_composition():
    """한글 조합 테스트"""
    print("\n📝 한글 조합 테스트")
    print("-" * 40)
    
    from korean_composition_algorithm import KoreanComposition
    
    composer = KoreanComposition()
    
    # 테스트 시퀀스
    test_sequences = [
        ['ㅎ', 'ㅏ', 'ㄴ'],  # 한
        ['ㄱ', 'ㅡ', 'ㄹ'],  # 글
        ['ㅅ', 'ㅏ', 'ㄹ'],  # 살
        ['ㅏ', 'ㅇ'],        # 앙
    ]
    
    for i, sequence in enumerate(test_sequences, 1):
        print(f"테스트 {i}: {sequence}")
        
        for char in sequence:
            result = composer.add_character(char)
            print(f"   '{char}' → {result.get('message', '처리됨')}")
        
        syllable = composer.complete_syllable()
        print(f"   ✅ 완성: '{syllable}'")
        
        composer.clear_composition()
    
    print("✅ 한글 조합 테스트 완료")
    
    return True

def main():
    """메인 테스트 함수"""
    print("🧪 SignGlove 추론 시스템 테스트 시작")
    print("=" * 60)
    
    tests = [
        ("추론 엔진", test_inference_engine),
        ("데이터 버퍼", test_data_buffer),
        ("한글 조합", test_korean_composition),
        ("통합 시스템", test_integrated_system),
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
        print("🎉 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print("⚠️ 일부 테스트가 실패했습니다.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
