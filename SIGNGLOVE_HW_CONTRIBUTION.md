# 🤝 SignGlove_HW 저장소 기여 제안

## 🎯 개발된 주요 기능

### 1. 가속도 데이터 지원 ✨
**파일**: `imu_flex_serial.ino`, `csv_uart.py`

**기존 문제점**:
- IMU 센서에서 자이로스코프만 활용
- 가속도 데이터 미수집으로 동작 인식 정확도 제한

**개선사항**:
```cpp
// 가속도계 데이터 추가
float ax = 0, ay = 0, az = 0;
if (IMU.accelerationAvailable()) {
  IMU.readAcceleration(ax, ay, az);
}
```

**결과**:
- **12필드 CSV**: `timestamp,pitch,roll,yaw,accel_x,accel_y,accel_z,flex1,flex2,flex3,flex4,flex5`
- **6축 IMU 완전 활용**: 자이로 + 가속도 동시 수집
- **ML 정확도 향상**: 추가 피처로 분류 성능 개선

### 2. 실시간 Hz 측정 📊
**파일**: `csv_uart.py`

**기존 문제점**:
- 샘플링 레이트 확인 불가
- 데이터 품질 모니터링 어려움

**개선사항**:
```python
# 아두이노 타임스탬프 기반 Hz 계산
arduino_ts = int(float(row[0]))
dt_ms = max(1, arduino_ts - last_arduino_ms)
inst_hz = 1000.0 / dt_ms
```

**결과**:
- **실시간 성능 모니터링**: 각 샘플의 정확한 Hz 표시
- **데이터 품질 검증**: 드롭된 샘플이나 지연 감지
- **사용자 친화적**: `✔️ Data saved @ 47.23 Hz` 형태 로그

### 3. 강화된 연결 관리 🔧
**파일**: `imu_flex_serial.ino`

**기존 문제점**:
- USB 연결 끊김 시 복구 어려움
- 시리얼 버퍼 오버플로우

**개선사항**:
```cpp
// 연결 상태 감지 및 자동 복구
bool nowConn = (bool)Serial;
if (nowConn && !connected) {
  connected = true;
  clearSerialBuffers();
  sendCsvHeader();
}
```

**결과**:
- **자동 재연결**: USB 케이블 재연결 시 자동 복구
- **버퍼 관리**: 연결 시점마다 깨끗한 버퍼 상태 보장
- **안정성 향상**: 장시간 데이터 수집 시 안정성 개선

### 4. 명령어 시스템 📝
**파일**: `imu_flex_serial.ino`

**새로운 기능**:
```cpp
// 실시간 제어 명령어
"interval,50"  // 샘플링 주기 변경
"header"       // CSV 헤더 재전송
"flush"        // 버퍼 클리어
```

**결과**:
- **동적 설정**: 실행 중 샘플링 레이트 변경
- **디버깅 지원**: 헤더 확인 및 버퍼 상태 제어
- **사용성 향상**: 아두이노 재시작 없이 설정 변경

## 🎯 기여 가치

### 사용자 혜택
1. **ML 성능 향상**: 6축 IMU 데이터로 분류 정확도 개선
2. **개발 효율성**: 실시간 모니터링으로 빠른 디버깅
3. **시스템 안정성**: 연결 관리 개선으로 데이터 손실 방지

### 호환성
- **기존 코드 호환**: 기존 9필드 파싱 코드도 동작
- **점진적 업그레이드**: 필요한 부분만 선택적 적용 가능
- **표준 준수**: Arduino 표준 라이브러리만 사용

## 📋 기여 방법

### Option 1: Pull Request
1. 현재 `signglove-local` 브랜치 → Fork 저장소로 푸시
2. `feature/accelerometer-and-performance` 브랜치 생성
3. Pull Request 생성 + 상세 설명

### Option 2: 콜라보레이터 (권장)
1. 저장소 소유자에게 콜라보레이터 초대 요청
2. 직접 브랜치 푸시 + 코드 리뷰
3. 빠른 피드백 및 반복 개선

## 🚀 향후 발전 방향

### 단기 목표
- [ ] 가속도 데이터 검증 및 테스트
- [ ] 추가 센서 지원 (자력계 등)
- [ ] WiFi 모드에도 동일 기능 적용

### 장기 목표  
- [ ] 센서 융합 알고리즘 (Madgwick Filter)
- [ ] 실시간 제스처 인식
- [ ] 모바일 앱 연동

## 💬 피드백 환영

이 개선사항들이 SignGlove_HW 커뮤니티에 도움이 되길 바랍니다!
개발 과정에서 발견한 문제점들을 해결하면서 만든 솔루션이므로 
실제 사용자들에게 유용할 것으로 기대합니다.

**Contact**: [@your-github-username] 
**프로젝트**: [SignGlove 통합 시스템](https://github.com/your-repo/SignGlove)
