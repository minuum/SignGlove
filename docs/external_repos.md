# 외부 저장소(서브모듈) 운영 가이드

본 문서는 `external/KLP-SignGlove`, `external/SignGlove_HW` 두 저장소를 서브모듈로 운영하는 규칙과 워크플로를 정의합니다.

## 구조

```
external/
├── KLP-SignGlove     # 전처리/실시간 추론/TTS 관련 참고 구현
└── SignGlove_HW      # 하드웨어/CSV 수집 관련 참고 구현
```

## 클론 및 초기화

```bash
git clone --recurse-submodules <repository-url>
cd SignGlove
git submodule update --init --recursive
```

## 업데이트 정책

- 주기적으로 원격 변경 사항을 반영하려면:

```bash
git submodule update --remote --merge
```

- 특정 커밋/태그로 버전 고정(핀)하여 재현성을 유지합니다.

```bash
cd external/KLP-SignGlove
git fetch origin
git checkout <commit-or-tag>
cd ../../
git add external/KLP-SignGlove
git commit -m "서브모듈: KLP-SignGlove 버전 고정(<commit-or-tag>)"
```

## 변경 관리 원칙

1. 서브모듈 내부 코드는 직접 수정하지 않습니다.
2. 변경이 필요하면 해당 저장소를 포크 → 수정 → PR 제안합니다.
3. 우리 프로젝트에서 필요한 통합은 어댑터 레이어로 처리합니다.
   - 전처리: `server/preprocessing.py`
   - 추론: `server/inference_engine.py`
   - TTS: `server/tts_engine.py`
   - CSV 수집: `scripts/csv_data_collector.py`
4. 보안/라이선스 준수:
   - 외부 저장소의 라이선스 조건을 준수합니다.
   - 민감정보가 포함되지 않도록 `.gitmodules` 외 설정 관리에 유의합니다.

## 문제 해결

- 서브모듈이 비어있음: `git submodule update --init --recursive`
- 원격 최신 반영 안 됨: `git submodule update --remote --merge`
- 충돌 발생: 상위 저장소에서 서브모듈 경로만 스테이징 후 커밋


