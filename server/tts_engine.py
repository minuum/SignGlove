#!/usr/bin/env python3
"""
SignGlove TTS (Text-to-Speech) 엔진
KLP-SignGlove의 한국어 음성 합성 기능 통합

포함 기능:
- 한국어 수어 → 음성 변환
- 신뢰도 기반 출력 제어
- 비동기 음성 합성
- 플랫폼별 TTS 엔진 지원 (macOS, Windows, Linux)
"""

import asyncio
import platform
import subprocess
import threading
import time
from typing import Dict, Optional, Callable, List
import logging
from datetime import datetime
from dataclasses import dataclass
from queue import Queue, Empty
import tempfile
import os

logger = logging.getLogger(__name__)


@dataclass
class TTSConfig:
    """TTS 설정 클래스"""
    voice: str = "auto"  # 음성 선택
    rate: int = 180      # 발화 속도 (words per minute)
    volume: float = 0.8  # 음량 (0.0-1.0)
    language: str = "ko" # 언어 코드
    enabled: bool = True # TTS 활성화 여부


@dataclass
class TTSRequest:
    """TTS 요청 클래스"""
    text: str
    confidence: float
    timestamp: datetime
    priority: int = 1  # 우선순위 (1=높음, 5=낮음)
    

class KoreanTTSEngine:
    """한국어 TTS 엔진 (KLP-SignGlove 기법)"""
    
    def __init__(self, config: TTSConfig = None):
        """
        TTS 엔진 초기화
        
        Args:
            config: TTS 설정
        """
        self.config = config or TTSConfig()
        self.platform = platform.system().lower()
        self.is_running = False
        self.request_queue = Queue()
        
        # 한국어 수어 → 음성 매핑
        self.ksl_to_speech = {
            # 자음
            'ㄱ': '기역',
            'ㄴ': '니은', 
            'ㄷ': '디귿',
            'ㄹ': '리을',
            'ㅁ': '미음',
            'ㅂ': '비읍',
            'ㅅ': '시옷',
            'ㅇ': '이응',
            'ㅈ': '지읒',
            'ㅊ': '치읓',
            'ㅋ': '키읔',
            'ㅌ': '티읕',
            'ㅍ': '피읖',
            'ㅎ': '히읗',
            
            # 모음
            'ㅏ': '아',
            'ㅑ': '야',
            'ㅓ': '어',
            'ㅕ': '여',
            'ㅗ': '오',
            'ㅛ': '요',
            'ㅜ': '우',
            'ㅠ': '유',
            'ㅡ': '으',
            'ㅣ': '이',
            
            # 숫자
            '0': '영',
            '1': '일',
            '2': '이',
            '3': '삼',
            '4': '사',
            '5': '오',
            '6': '육',
            '7': '칠',
            '8': '팔',
            '9': '구',
            
            # 특수 케이스
            'ERROR': '인식 오류',
            'UNKNOWN': '알 수 없음'
        }
        
        # 플랫폼별 초기화
        self._initialize_platform_tts()
        
        logger.info(f"TTS 엔진 초기화: {self.platform} 플랫폼")
    
    def _initialize_platform_tts(self):
        """플랫폼별 TTS 초기화"""
        try:
            if self.platform == "darwin":  # macOS
                self._initialize_macos_tts()
            elif self.platform == "windows":
                self._initialize_windows_tts()
            elif self.platform == "linux":
                self._initialize_linux_tts()
            else:
                logger.warning(f"지원하지 않는 플랫폼: {self.platform}")
                self.config.enabled = False
        except Exception as e:
            logger.error(f"TTS 초기화 실패: {e}")
            self.config.enabled = False
    
    def _initialize_macos_tts(self):
        """macOS TTS 초기화"""
        try:
            # 사용 가능한 한국어 음성 확인
            result = subprocess.run(['say', '-v', '?'], capture_output=True, text=True)
            voices = result.stdout
            
            # 한국어 음성 찾기
            korean_voices = []
            for line in voices.split('\n'):
                if 'ko_KR' in line or 'Korean' in line:
                    voice_name = line.split()[0]
                    korean_voices.append(voice_name)
            
            if korean_voices:
                self.config.voice = korean_voices[0]  # 첫 번째 한국어 음성 사용
                logger.info(f"macOS 한국어 음성 설정: {self.config.voice}")
            else:
                logger.warning("한국어 음성을 찾을 수 없습니다. 기본 음성 사용")
                self.config.voice = "Yuna"  # 기본 한국어 음성
                
        except Exception as e:
            logger.error(f"macOS TTS 초기화 오류: {e}")
            self.config.voice = "Yuna"
    
    def _initialize_windows_tts(self):
        """Windows TTS 초기화"""
        try:
            # Windows SAPI 음성 확인
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            voices = speaker.GetVoices()
            
            # 한국어 음성 찾기
            for voice in voices:
                if 'korean' in voice.GetDescription().lower() or 'ko-kr' in voice.GetDescription().lower():
                    self.config.voice = voice.GetDescription()
                    break
            
            logger.info(f"Windows 한국어 음성 설정: {self.config.voice}")
            
        except ImportError:
            logger.warning("pywin32가 설치되지 않았습니다. 기본 음성 사용")
        except Exception as e:
            logger.error(f"Windows TTS 초기화 오류: {e}")
    
    def _initialize_linux_tts(self):
        """Linux TTS 초기화"""
        try:
            # espeak 또는 festival 확인
            espeak_available = subprocess.run(['which', 'espeak'], capture_output=True).returncode == 0
            festival_available = subprocess.run(['which', 'festival'], capture_output=True).returncode == 0
            
            if espeak_available:
                self.tts_command = "espeak"
                logger.info("Linux TTS: espeak 사용")
            elif festival_available:
                self.tts_command = "festival"
                logger.info("Linux TTS: festival 사용")
            else:
                logger.error("Linux TTS 엔진을 찾을 수 없습니다 (espeak, festival)")
                self.config.enabled = False
                
        except Exception as e:
            logger.error(f"Linux TTS 초기화 오류: {e}")
            self.config.enabled = False
    
    def convert_ksl_to_speech(self, ksl_text: str) -> str:
        """
        KSL 텍스트를 음성용 텍스트로 변환
        
        Args:
            ksl_text: KSL 텍스트 (예: "ㄱ", "ㅏ", "1")
            
        Returns:
            speech_text: 음성용 텍스트 (예: "기역", "아", "일")
        """
        if ksl_text in self.ksl_to_speech:
            return self.ksl_to_speech[ksl_text]
        else:
            # 복합 단어인 경우 글자별 분리
            result = []
            for char in ksl_text:
                if char in self.ksl_to_speech:
                    result.append(self.ksl_to_speech[char])
                else:
                    result.append(char)  # 한글 완성형은 그대로
            
            return ' '.join(result) if result else ksl_text
    
    def _synthesize_speech_macos(self, text: str) -> bool:
        """macOS에서 음성 합성"""
        try:
            cmd = ['say', '-v', self.config.voice, '-r', str(self.config.rate)]
            cmd.append(text)
            
            subprocess.run(cmd, check=True)
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"macOS TTS 오류: {e}")
            return False
    
    def _synthesize_speech_windows(self, text: str) -> bool:
        """Windows에서 음성 합성"""
        try:
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            
            # 음성 속도 설정
            speaker.Rate = (self.config.rate - 180) // 20  # SAPI rate range: -10 to 10
            
            # 음량 설정
            speaker.Volume = int(self.config.volume * 100)
            
            speaker.Speak(text)
            return True
            
        except Exception as e:
            logger.error(f"Windows TTS 오류: {e}")
            return False
    
    def _synthesize_speech_linux(self, text: str) -> bool:
        """Linux에서 음성 합성"""
        try:
            if hasattr(self, 'tts_command'):
                if self.tts_command == "espeak":
                    cmd = ['espeak', '-s', str(self.config.rate), '-a', str(int(self.config.volume * 100))]
                    cmd.append(text)
                    subprocess.run(cmd, check=True)
                elif self.tts_command == "festival":
                    # festival은 임시 파일 방식 사용
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                        f.write(text)
                        temp_file = f.name
                    
                    subprocess.run(['festival', '--tts', temp_file], check=True)
                    os.unlink(temp_file)
                
                return True
            else:
                logger.error("Linux TTS 명령어가 설정되지 않았습니다.")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Linux TTS 오류: {e}")
            return False
    
    def synthesize_speech(self, text: str) -> bool:
        """
        음성 합성 실행
        
        Args:
            text: 합성할 텍스트
            
        Returns:
            success: 합성 성공 여부
        """
        if not self.config.enabled:
            logger.warning("TTS가 비활성화되어 있습니다.")
            return False
        
        logger.info(f"TTS 합성: {text}")
        
        try:
            if self.platform == "darwin":
                return self._synthesize_speech_macos(text)
            elif self.platform == "windows":
                return self._synthesize_speech_windows(text)
            elif self.platform == "linux":
                return self._synthesize_speech_linux(text)
            else:
                logger.error(f"지원하지 않는 플랫폼: {self.platform}")
                return False
                
        except Exception as e:
            logger.error(f"TTS 합성 오류: {e}")
            return False
    
    def speak_ksl(self, ksl_text: str, confidence: float = 1.0) -> bool:
        """
        KSL 텍스트 음성 출력
        
        Args:
            ksl_text: KSL 텍스트
            confidence: 신뢰도 (0.0-1.0)
            
        Returns:
            success: 출력 성공 여부
        """
        # 신뢰도 체크
        if confidence < 0.7:  # 신뢰도가 낮으면 출력하지 않음
            logger.debug(f"신뢰도가 낮아 TTS 스킵: {ksl_text} (신뢰도: {confidence:.2f})")
            return False
        
        # KSL → 음성 텍스트 변환
        speech_text = self.convert_ksl_to_speech(ksl_text)
        
        # 음성 합성
        return self.synthesize_speech(speech_text)
    
    def add_request(self, text: str, confidence: float, priority: int = 1):
        """TTS 요청 추가 (비동기 처리용)"""
        request = TTSRequest(
            text=text,
            confidence=confidence,
            timestamp=datetime.now(),
            priority=priority
        )
        
        try:
            self.request_queue.put_nowait(request)
        except:
            logger.warning("TTS 요청 큐 오버플로우")
    
    def _tts_worker(self):
        """TTS 워커 스레드"""
        while self.is_running:
            try:
                request = self.request_queue.get(timeout=1.0)
                self.speak_ksl(request.text, request.confidence)
                self.request_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"TTS 워커 오류: {e}")
    
    def start(self):
        """TTS 엔진 시작"""
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.worker_thread.start()
        
        logger.info("TTS 엔진 시작")
    
    def stop(self):
        """TTS 엔진 중지"""
        self.is_running = False
        
        if hasattr(self, 'worker_thread'):
            self.worker_thread.join(timeout=2.0)
        
        logger.info("TTS 엔진 중지")
    
    def test_speech(self):
        """TTS 테스트"""
        test_phrases = ["안녕하세요", "기역", "아", "일"]
        
        for phrase in test_phrases:
            logger.info(f"TTS 테스트: {phrase}")
            success = self.synthesize_speech(phrase)
            if not success:
                logger.error(f"TTS 테스트 실패: {phrase}")
                return False
            time.sleep(1)
        
        logger.info("TTS 테스트 완료")
        return True


# 전역 TTS 엔진 인스턴스
_global_tts_engine: Optional[KoreanTTSEngine] = None


def get_tts_engine() -> KoreanTTSEngine:
    """전역 TTS 엔진 인스턴스 반환"""
    global _global_tts_engine
    
    if _global_tts_engine is None:
        _global_tts_engine = KoreanTTSEngine()
    
    return _global_tts_engine


def speak_ksl_async(ksl_text: str, confidence: float = 1.0):
    """비동기 KSL 음성 출력"""
    engine = get_tts_engine()
    engine.add_request(ksl_text, confidence)


def test_tts():
    """TTS 테스트 함수"""
    engine = get_tts_engine()
    return engine.test_speech()
