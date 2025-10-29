#!/usr/bin/env python3
"""
한글 조합 알고리즘
- 초성, 중성, 종성을 조합하여 한글 음절 생성
- 실시간 한글 입력 시스템을 위한 핵심 모듈
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class CompositionState:
    """조합 상태"""
    current_chosung: Optional[str] = None
    current_jungsung: Optional[str] = None
    current_jongsung: Optional[str] = None
    current_syllable: str = ""
    current_word: str = ""
    composition_buffer: List[str] = None
    
    def __post_init__(self):
        if self.composition_buffer is None:
            self.composition_buffer = []

class KoreanComposition:
    """한글 조합 클래스"""
    
    # 한글 유니코드 범위
    HANGUL_START = 0xAC00  # '가'
    HANGUL_END = 0xD7A3    # '힣'
    
    # 초성 (19개)
    CHOSUNG = [
        'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 
        'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
    ]
    
    # 중성 (21개)
    JUNGSUNG = [
        'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 
        'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
    ]
    
    # 종성 (28개, 첫 번째는 없음)
    JONGSUNG = [
        '', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 
        'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 
        'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
    ]
    
    def __init__(self):
        self.current_chosung = None
        self.current_jungsung = None
        self.current_jongsung = None
        self.composition_buffer = []
        self.word_buffer = []
    
    def is_consonant(self, char: str) -> bool:
        """자음인지 확인"""
        return char in self.CHOSUNG
    
    def is_vowel(self, char: str) -> bool:
        """모음인지 확인"""
        return char in self.JUNGSUNG
    
    def get_chosung_index(self, char: str) -> int:
        """초성 인덱스 반환"""
        try:
            return self.CHOSUNG.index(char)
        except ValueError:
            return -1
    
    def get_jungsung_index(self, char: str) -> int:
        """중성 인덱스 반환"""
        try:
            return self.JUNGSUNG.index(char)
        except ValueError:
            return -1
    
    def get_jongsung_index(self, char: str) -> int:
        """종성 인덱스 반환"""
        try:
            return self.JONGSUNG.index(char)
        except ValueError:
            return -1
    
    def compose_hangul(self, chosung: str, jungsung: str, jongsung: str = '') -> str:
        """한글 음절 조합"""
        chosung_idx = self.get_chosung_index(chosung)
        jungsung_idx = self.get_jungsung_index(jungsung)
        jongsung_idx = self.get_jongsung_index(jongsung)
        
        if chosung_idx == -1 or jungsung_idx == -1:
            return ''
        
        # 한글 유니코드 계산
        hangul_code = self.HANGUL_START + (chosung_idx * 21 + jungsung_idx) * 28 + jongsung_idx
        return chr(hangul_code)
    
    def add_character(self, char: str) -> Dict[str, Any]:
        """문자 추가 및 조합"""
        result = {
            'char': char,
            'current_syllable': '',
            'current_word': '',
            'can_complete': False,
            'composition_state': self.get_composition_state(),
            'message': ""
        }
        
        if self.is_consonant(char):
            # 자음 입력
            if self.current_chosung is None:
                # 초성으로 설정
                self.current_chosung = char
                result['message'] = f"초성 '{char}' 설정"
            elif self.current_jungsung is None:
                # 초성 변경
                self.current_chosung = char
                result['message'] = f"초성 '{char}' 변경"
            elif self.current_jongsung is None:
                # 종성으로 설정
                self.current_jongsung = char
                result['message'] = f"종성 '{char}' 설정"
            else:
                # 새로운 음절 시작
                self.complete_syllable()
                self.current_chosung = char
                self.current_jungsung = None
                self.current_jongsung = None
                result['message'] = f"새 음절 시작, 초성 '{char}'"
                
        elif self.is_vowel(char):
            # 모음 입력
            if self.current_chosung is None:
                # 초성이 없는 경우, 모음만으로는 조합 불가
                result['message'] = f"초성이 필요합니다. '{char}' 무시됨"
                return result
            elif self.current_jungsung is None:
                # 중성으로 설정
                self.current_jungsung = char
                result['message'] = f"중성 '{char}' 설정"
            else:
                # 이미 중성이 있는 경우, 새 음절 시작
                self.complete_syllable()
                self.current_chosung = 'ㅇ'  # ㅇ 초성으로 시작
                self.current_jungsung = char
                self.current_jongsung = None
                result['message'] = f"새 음절 시작, 중성 '{char}'"
                
        # 현재 음절 조합
        if self.current_chosung and self.current_jungsung:
            syllable = self.compose_hangul(
                self.current_chosung, 
                self.current_jungsung, 
                self.current_jongsung or ''
            )
            result['current_syllable'] = syllable
            result['can_complete'] = True
            
        # 현재 단어 조합
        current_word = ''.join(self.word_buffer) + result['current_syllable']
        result['current_word'] = current_word
        
        return result
    
    def complete_syllable(self) -> str:
        """현재 음절 완성"""
        if self.current_chosung and self.current_jungsung:
            syllable = self.compose_hangul(
                self.current_chosung, 
                self.current_jungsung, 
                self.current_jongsung or ''
            )
            self.composition_buffer.append(syllable)
            self.word_buffer.append(syllable)
            
            # 상태 초기화
            self.current_chosung = None
            self.current_jungsung = None
            self.current_jongsung = None
            
            return syllable
        return ''
    
    def complete_word(self) -> str:
        """현재 단어 완성"""
        # 마지막 음절 완성
        last_syllable = self.complete_syllable()
        
        if last_syllable:
            word = ''.join(self.word_buffer)
            self.clear_composition()
            return word
        return ''
    
    def clear_composition(self):
        """조합 상태 초기화"""
        self.current_chosung = None
        self.current_jungsung = None
        self.current_jongsung = None
        self.composition_buffer.clear()
        self.word_buffer.clear()
    
    def get_composition_state(self) -> Dict[str, Any]:
        """현재 조합 상태 반환"""
        return {
            'current_chosung': self.current_chosung,
            'current_jungsung': self.current_jungsung,
            'current_jongsung': self.current_jongsung,
            'current_syllable': self.compose_hangul(
                self.current_chosung or '', 
                self.current_jungsung or '', 
                self.current_jongsung or ''
            ) if self.current_chosung and self.current_jungsung else '',
            'current_word': ''.join(self.word_buffer),
            'composition_buffer': self.composition_buffer.copy()
        }

def main():
    """테스트 함수"""
    print("한글 조합 알고리즘 테스트")
    
    composer = KoreanComposition()
    
    # 테스트 시퀀스
    test_sequences = [
        ['ㅎ', 'ㅏ', 'ㄴ'],  # 한
        ['ㄱ', 'ㅡ', 'ㄹ'],  # 글
        ['ㅅ', 'ㅏ', 'ㄹ'],  # 살
        ['ㅏ', 'ㅇ'],        # 앙
    ]
    
    for i, sequence in enumerate(test_sequences, 1):
        print(f"\n테스트 {i}: {sequence}")
        
        for char in sequence:
            result = composer.add_character(char)
            print(f"   '{char}' → {result.get('message', '처리됨')}")
        
        syllable = composer.complete_syllable()
        print(f"   ✅ 완성: '{syllable}'")
        
        composer.clear_composition()

if __name__ == "__main__":
    main()
