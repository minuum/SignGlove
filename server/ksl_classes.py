"""
한국어 수어 클래스 정의 모듈
Korean Sign Language (KSL) 클래스 정의 및 관리 시스템

담당: YUBEEN, 정재연
작성: 이민우 (기본 틀 제공)
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json


class KSLCategory(Enum):
    """한국어 수어 카테고리 분류"""
    VOWEL = "vowel"           # 모음 (ㅏ, ㅓ, ㅗ, ㅜ, ㅡ, ㅣ, ㅑ, ㅕ, ㅛ, ㅠ)  
    CONSONANT = "consonant"   # 자음 (ㄱ, ㄴ, ㄷ, ㄹ, ㅁ, ㅂ, ㅅ, ㅇ, ㅈ, ㅊ, ㅋ, ㅌ, ㅍ, ㅎ)
    NUMBER = "number"         # 숫자 (0-9)
    WORD = "word"             # 단어 (안녕하세요, 감사합니다 등)
    ALPHABET = "alphabet"     # 영어 알파벳 (A-Z)


@dataclass
class SensorPattern:
    """센서 패턴 정의 클래스"""
    flex_sensors: List[float]        # 플렉스 센서 값 [0-1023] * 5개
    gyroscope: Dict[str, float]      # 자이로스코프 값 {x, y, z, roll, pitch, yaw}
    accelerometer: Dict[str, float]  # 가속도계 값 {x, y, z}
    duration_ms: int                 # 지속 시간 (밀리초)
    confidence: float                # 신뢰도 [0.0-1.0]
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "flex_sensors": self.flex_sensors,
            "gyroscope": self.gyroscope,
            "accelerometer": self.accelerometer,
            "duration_ms": self.duration_ms,
            "confidence": self.confidence
        }


@dataclass
class KSLClass:
    """한국어 수어 클래스 정의"""
    name: str                        # 수어 이름 (예: "ㅏ", "안녕하세요")
    category: KSLCategory            # 카테고리
    description: str                 # 설명
    reference_pattern: SensorPattern # 기준 센서 패턴
    variations: List[SensorPattern]  # 변형 패턴들
    difficulty_level: int            # 난이도 (1-5)
    learning_priority: int           # 학습 우선순위 (1-10)
    
    # 메타데이터
    created_date: datetime
    updated_date: datetime
    created_by: str                  # 작성자
    validated: bool                  # 검증 완료 여부
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "reference_pattern": self.reference_pattern.to_dict(),
            "variations": [v.to_dict() for v in self.variations],
            "difficulty_level": self.difficulty_level,
            "learning_priority": self.learning_priority,
            "created_date": self.created_date.isoformat(),
            "updated_date": self.updated_date.isoformat(),
            "created_by": self.created_by,
            "validated": self.validated
        }


class KSLClassManager:
    """한국어 수어 클래스 관리자"""
    
    def __init__(self):
        self.classes: Dict[str, KSLClass] = {}
        self._initialize_basic_classes()
    
    def _initialize_basic_classes(self):
        """기본 수어 클래스들 초기화 (YUBEEN, 정재연님이 확장할 기본 틀)"""
        
        # === 기본 모음 클래스 템플릿 ===
        basic_vowels = [
            ("ㅏ", "아 모음 - 손바닥을 앞으로, 검지 위로"),
            ("ㅓ", "어 모음 - 손바닥을 몸 쪽으로, 검지 위로"),
            ("ㅗ", "오 모음 - 주먹을 쥐고 엄지를 위로"),
            ("ㅜ", "우 모음 - 주먹을 쥐고 엄지를 아래로"),
            ("ㅡ", "으 모음 - 손바닥을 아래로, 손가락 펼침"),
            ("ㅣ", "이 모음 - 검지만 세우고 위로")
        ]
        
        for vowel, description in basic_vowels:
            # TODO: YUBEEN, 정재연님이 실제 센서 값으로 교체
            template_pattern = SensorPattern(
                flex_sensors=[500, 500, 500, 500, 500],  # 임시 값
                gyroscope={"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
                accelerometer={"x": 0, "y": -1, "z": 0},  # 기본 중력 방향
                duration_ms=1000,
                confidence=0.8
            )
            
            ksl_class = KSLClass(
                name=vowel,
                category=KSLCategory.VOWEL,
                description=description,
                reference_pattern=template_pattern,
                variations=[],  # TODO: 변형 패턴 추가
                difficulty_level=2,
                learning_priority=8,
                created_date=datetime.now(),
                updated_date=datetime.now(),
                created_by="system_template",
                validated=False  # TODO: 실제 데이터로 검증 필요
            )
            
            self.classes[vowel] = ksl_class
        
        # === 기본 자음 클래스 템플릿 ===
        basic_consonants = [
            ("ㄱ", "기역 - 주먹을 쥐고 검지 구부림"),
            ("ㄴ", "니은 - 검지와 중지만 펼침"),
            ("ㄷ", "디귿 - 엄지와 검지만 펼침"),
            ("ㄹ", "리을 - 검지를 구부려 갈고리 모양"),
            ("ㅁ", "미음 - 주먹을 쥐고 엄지를 옆으로")
        ]
        
        for consonant, description in basic_consonants:
            template_pattern = SensorPattern(
                flex_sensors=[300, 300, 300, 300, 300],  # 임시 값
                gyroscope={"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
                accelerometer={"x": 0, "y": -1, "z": 0},
                duration_ms=800,
                confidence=0.8
            )
            
            ksl_class = KSLClass(
                name=consonant,
                category=KSLCategory.CONSONANT,
                description=description,
                reference_pattern=template_pattern,
                variations=[],
                difficulty_level=3,
                learning_priority=7,
                created_date=datetime.now(),
                updated_date=datetime.now(),
                created_by="system_template",
                validated=False
            )
            
            self.classes[consonant] = ksl_class
        
        # === 기본 숫자 클래스 템플릿 ===
        for i in range(10):
            description = f"숫자 {i} - " + ["주먹", "검지", "검지+중지", "검지+중지+약지", "엄지 제외 4개", 
                                         "5개 펼침", "엄지+새끼", "엄지+검지+중지", "엄지+검지", "검지 구부림"][i]
            
            template_pattern = SensorPattern(
                flex_sensors=[400, 400, 400, 400, 400],
                gyroscope={"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
                accelerometer={"x": 0, "y": -1, "z": 0},
                duration_ms=600,
                confidence=0.9
            )
            
            ksl_class = KSLClass(
                name=str(i),
                category=KSLCategory.NUMBER,
                description=description,
                reference_pattern=template_pattern,
                variations=[],
                difficulty_level=1,
                learning_priority=10,
                created_date=datetime.now(),
                updated_date=datetime.now(),
                created_by="system_template", 
                validated=False
            )
            
            self.classes[str(i)] = ksl_class
    
    def add_class(self, ksl_class: KSLClass) -> bool:
        """새로운 수어 클래스 추가"""
        if ksl_class.name in self.classes:
            return False  # 이미 존재함
        
        self.classes[ksl_class.name] = ksl_class
        return True
    
    def update_class(self, name: str, **kwargs) -> bool:
        """수어 클래스 업데이트"""
        if name not in self.classes:
            return False
        
        ksl_class = self.classes[name]
        for key, value in kwargs.items():
            if hasattr(ksl_class, key):
                setattr(ksl_class, key, value)
        
        ksl_class.updated_date = datetime.now()
        return True
    
    def get_class(self, name: str) -> Optional[KSLClass]:
        """수어 클래스 조회"""
        return self.classes.get(name)
    
    def get_classes_by_category(self, category: KSLCategory) -> List[KSLClass]:
        """카테고리별 수어 클래스 조회"""
        return [cls for cls in self.classes.values() if cls.category == category]
    
    def get_validated_classes(self) -> List[KSLClass]:
        """검증된 수어 클래스만 조회"""
        return [cls for cls in self.classes.values() if cls.validated]
    
    def get_classes_by_priority(self, min_priority: int = 5) -> List[KSLClass]:
        """우선순위별 수어 클래스 조회"""
        return sorted(
            [cls for cls in self.classes.values() if cls.learning_priority >= min_priority],
            key=lambda x: x.learning_priority,
            reverse=True
        )
    
    def export_to_json(self, filepath: str) -> bool:
        """JSON 파일로 내보내기"""
        try:
            data = {
                "classes": {name: cls.to_dict() for name, cls in self.classes.items()},
                "metadata": {
                    "total_classes": len(self.classes),
                    "categories": {cat.value: len(self.get_classes_by_category(cat)) 
                                 for cat in KSLCategory},
                    "validated_count": len(self.get_validated_classes()),
                    "export_date": datetime.now().isoformat()
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception:
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """수어 클래스 통계 정보"""
        return {
            "total_classes": len(self.classes),
            "by_category": {
                cat.value: len(self.get_classes_by_category(cat)) 
                for cat in KSLCategory
            },
            "validated_classes": len(self.get_validated_classes()),
            "avg_difficulty": sum(cls.difficulty_level for cls in self.classes.values()) / len(self.classes),
            "high_priority": len(self.get_classes_by_priority(8))
        }


# 전역 KSL 클래스 관리자
ksl_manager = KSLClassManager()


def get_ksl_class(name: str) -> Optional[KSLClass]:
    """수어 클래스 조회 편의 함수"""
    return ksl_manager.get_class(name)


def get_all_ksl_classes() -> Dict[str, KSLClass]:
    """모든 수어 클래스 조회"""
    return ksl_manager.classes.copy()


# === YUBEEN & 정재연님을 위한 작업 가이드 ===
"""
TODO: 다음 작업들을 수행해주세요

1. 실제 센서 데이터 수집:
   - 각 수어 동작을 수행하면서 실제 센서 값 측정
   - reference_pattern의 임시 값들을 실제 값으로 교체
   
2. 변형 패턴 추가:
   - 같은 수어라도 사람마다 다른 패턴 존재
   - variations 리스트에 여러 패턴 추가
   
3. 고급 수어 추가:
   - 복합 모음: ㅐ, ㅔ, ㅚ, ㅟ, ㅢ 등
   - 복합 자음: ㄲ, ㄸ, ㅃ, ㅆ, ㅉ 등
   - 일반 단어: "안녕하세요", "감사합니다" 등
   
4. 검증 및 테스트:
   - validated = True로 변경하기 전 충분한 테스트
   - 다양한 사용자에게 패턴 검증 받기
   
5. 우선순위 조정:
   - learning_priority 값을 실제 학습 중요도에 따라 조정
   - 기초 → 중급 → 고급 순서로 설정

사용 예시:
```python
# 새로운 수어 클래스 추가
new_vowel = KSLClass(
    name="ㅐ",
    category=KSLCategory.VOWEL,
    description="애 모음 - 검지와 중지를 벌려서 V자",
    reference_pattern=SensorPattern(...),  # 실제 센서 값
    # ... 기타 필드들
)
ksl_manager.add_class(new_vowel)

# 기존 클래스 업데이트
ksl_manager.update_class("ㅏ", validated=True, reference_pattern=new_pattern)
```
""" 