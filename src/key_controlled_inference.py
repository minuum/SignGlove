#!/usr/bin/env python3
"""
키 제어 추론 시스템
- 키보드 입력을 통한 추론 제어
- 실시간 한글 조합과 연동
"""

import sys
import os
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass

# 현재 디렉토리의 모듈 import
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from korean_composition_algorithm import KoreanComposition

# OS별 키보드 입력 모듈 임포트
if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty

class InferenceState(Enum):
    """추론 상태"""
    IDLE = "대기"
    COLLECTING = "데이터 수집"
    INFERRING = "추론 중"
    COMPOSING = "조합 중"

@dataclass
class InferenceResult:
    """추론 결과"""
    predicted_class: str
    confidence: float
    timestamp: float
    state: InferenceState
    composition_result: Optional[Dict] = None

class AdvancedKeyController:
    """고급 키 제어 시스템"""
    
    def __init__(self, composition: KoreanComposition):
        self.composition = composition
        self.running = False
        self.inference_active = False
        self.current_state = InferenceState.IDLE
        
        # 키 핸들러
        self.default_handlers = {}
        self.custom_handlers = {}
        
        # 통계
        self.stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'completed_words': 0,
            'completed_syllables': 0,
            'session_start': time.time()
        }
        
        self.setup_default_handlers()
        print("✅ 키 제어 시스템 초기화 완료")
    
    def setup_default_handlers(self):
        """기본 핸들러 설정"""
        self.default_handlers = {
            'space': self._handle_space,
            'enter': self._handle_enter,
            'esc': self._handle_esc,
            'f1': self._toggle_inference,
            'f2': self._clear_composition,
            'f3': self._show_stats,
            'h': self._show_help
        }
    
    def add_custom_handler(self, key: str, handler: Callable, description: str = ""):
        """커스텀 핸들러 추가"""
        self.custom_handlers[key] = {
            'handler': handler,
            'description': description
        }
        print(f"✅ 커스텀 핸들러 추가: {key} - {description}")
    
    def remove_custom_handler(self, key: str):
        """커스텀 핸들러 제거"""
        if key in self.custom_handlers:
            del self.custom_handlers[key]
            print(f"✅ 커스텀 핸들러 제거: {key}")
    
    def start_inference(self):
        """추론 시스템 시작"""
        self.running = True
        self.inference_active = True
        print("🚀 키 제어 추론 시스템 시작")
        print("💡 키보드 입력을 기다리는 중...")
        
        try:
            while self.running:
                key = self.get_key()
                if key:
                    self.handle_key(key)
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n⏹️ 사용자에 의해 중단됨")
        finally:
            self.stop_inference()
    
    def stop_inference(self):
        """추론 시스템 중지"""
        self.running = False
        self.inference_active = False
        print("🔚 키 제어 추론 시스템 중지")
    
    def get_key(self) -> str:
        """키 입력 받기"""
        if sys.platform == 'win32':
            if msvcrt.kbhit():
                try:
                    return msvcrt.getch().decode('utf-8').lower()
                except UnicodeDecodeError:
                    return ''
            return ""
        else:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                import select
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    ch = sys.stdin.read(1)
                    return ch.lower()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ""
    
    def handle_key(self, key: str):
        """키 처리"""
        # 커스텀 핸들러 우선
        if key in self.custom_handlers:
            try:
                self.custom_handlers[key]['handler']()
            except Exception as e:
                print(f"❌ 커스텀 핸들러 오류 ({key}): {e}")
            return
        
        # 기본 핸들러
        if key in self.default_handlers:
            try:
                self.default_handlers[key]()
            except Exception as e:
                print(f"❌ 기본 핸들러 오류 ({key}): {e}")
        else:
            print(f"⚠️ 알 수 없는 키: {key}")
    
    def _handle_space(self):
        """SPACE 키: 추론 시작/중지"""
        if self.current_state == InferenceState.IDLE:
            self.current_state = InferenceState.COLLECTING
            print("📡 데이터 수집 시작")
        elif self.current_state == InferenceState.COLLECTING:
            self.current_state = InferenceState.INFERRING
            print("🔍 추론 실행")
        else:
            self.current_state = InferenceState.IDLE
            print("⏹️ 추론 중지")
    
    def _handle_enter(self):
        """ENTER 키: 음절 완성"""
        if self.current_state == InferenceState.COMPOSING:
            syllable = self.composition.complete_syllable()
            if syllable:
                print(f"✅ 음절 완성: '{syllable}'")
                self.stats['completed_syllables'] += 1
            else:
                print("❌ 완성할 음절이 없습니다")
        else:
            print("⚠️ 조합 모드가 아닙니다")
    
    def _handle_esc(self):
        """ESC 키: 종료"""
        print("👋 시스템 종료")
        self.running = False
    
    def _toggle_inference(self):
        """F1 키: 추론 토글"""
        self.inference_active = not self.inference_active
        status = "활성화" if self.inference_active else "비활성화"
        print(f"🔄 추론 시스템 {status}")
    
    def _clear_composition(self):
        """F2 키: 조합 초기화"""
        self.composition.clear_composition()
        print("🗑️ 조합 상태 초기화됨")
    
    def _show_stats(self):
        """F3 키: 통계 표시"""
        runtime = time.time() - self.stats['session_start']
        print(f"\n📊 추론 시스템 통계:")
        print(f"   실행 시간: {runtime:.1f}초")
        print(f"   총 예측: {self.stats['total_predictions']}개")
        print(f"   성공 예측: {self.stats['successful_predictions']}개")
        print(f"   완성된 단어: {self.stats['completed_words']}개")
        print(f"   완성된 음절: {self.stats['completed_syllables']}개")
        print(f"   현재 상태: {self.current_state.value}")
    
    def _show_help(self):
        """H 키: 도움말"""
        print("\n📋 키 제어 도움말:")
        print("   SPACE: 추론 시작/중지")
        print("   ENTER: 음절 완성")
        print("   F1: 추론 토글")
        print("   F2: 조합 초기화")
        print("   F3: 통계 표시")
        print("   H: 도움말")
        print("   ESC: 종료")
    
    def execute_inference(self, sensor_data: List[float]) -> InferenceResult:
        """추론 실행 (Mock)"""
        if not self.inference_active:
            return None
        
        # Mock 추론 결과
        import random
        classes = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                  'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        
        predicted_class = random.choice(classes)
        confidence = random.uniform(0.6, 0.95)
        
        # 한글 조합에 추가
        composition_result = self.composition.add_character(predicted_class)
        
        # 통계 업데이트
        self.stats['total_predictions'] += 1
        if confidence > 0.7:
            self.stats['successful_predictions'] += 1
        
        result = InferenceResult(
            predicted_class=predicted_class,
            confidence=confidence,
            timestamp=time.time(),
            state=self.current_state,
            composition_result=composition_result
        )
        
        return result

def main():
    """테스트 함수"""
    print("키 제어 추론 시스템 테스트")
    
    # 한글 조합기 생성
    composer = KoreanComposition()
    
    # 키 컨트롤러 생성
    controller = AdvancedKeyController(composer)
    
    # 커스텀 핸들러 추가
    def custom_test_handler():
        print("🎯 커스텀 핸들러 실행!")
    
    controller.add_custom_handler('t', custom_test_handler, "테스트 핸들러")
    
    # 시스템 시작
    controller.start_inference()

if __name__ == "__main__":
    main()
