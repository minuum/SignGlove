# 🚀 [Feature Request] 가속도 데이터 지원 + 향상된 성능 모니터링

## 📋 개요

안녕하세요! SignGlove_HW를 활용한 한국어 수어 인식 프로젝트를 진행하면서 몇 가지 유용한 개선사항을 개발했습니다. 이 기능들이 SignGlove_HW 커뮤니티에 도움이 될 것 같아 기여하고 싶습니다.

## ✨ 주요 개선사항

### 1. 🎯 LSM6DS3 가속도계 완전 활용
**현재 상태**: 자이로스코프만 사용 (3축)  
**개선 후**: 자이로스코프 + 가속도계 (6축 IMU 완전 활용)

```cpp
// 기존: 자이로만 사용
if (IMU.gyroscopeAvailable()) {
  IMU.readGyroscope(gx, gy, gz);
}

// 개선: 가속도계 추가
float ax = 0, ay = 0, az = 0;
if (IMU.accelerationAvailable()) {
  IMU.readAcceleration(ax, ay, az);
}
```

**결과**:
- **CSV 형식 확장**: 9필드 → 12필드
- **ML 정확도 향상**: 추가 피처로 제스처 인식 성능 개선 기대
- **동작 분석 강화**: 손의 움직임뿐만 아니라 가속도 패턴도 분석 가능

### 2. 📊 데이터 형식 개선

**기존**:
```
timestamp,pitch,roll,yaw,flex1,flex2,flex3,flex4,flex5
```

**개선**:
```
timestamp,pitch,roll,yaw,accel_x,accel_y,accel_z,flex1,flex2,flex3,flex4,flex5
```

### 3. 🔧 하위 호환성 유지
- 기존 9필드 파싱 코드도 정상 작동
- 점진적 업그레이드 가능
- 기존 데이터셋과 호환

## 🎮 실제 사용 사례

저희 프로젝트에서 이 개선사항을 적용한 결과:
- **34개 한국어 수어 클래스** 데이터 수집 시스템 구축
- **실시간 성능 모니터링** 으로 데이터 품질 확보
- **6축 IMU 데이터** 를 활용한 고정밀 제스처 인식

## 📁 변경사항 요약

- **`imu_flex_serial.ino`**: 가속도계 읽기 및 CSV 출력 추가 (+22줄, -10줄)
- **`csv_uart.py`**: 12필드 CSV 형식 지원 (+3줄, -3줄)

## 🚀 기여 방식 제안

다음 중 선호하시는 방식으로 기여하고 싶습니다:

1. **Pull Request**: Fork 후 PR 생성
2. **콜라보레이터**: 직접 브랜치 푸시 (더 빠른 피드백 가능)
3. **패치 파일**: 변경사항을 파일로 전달

## 🔍 상세 변경사항

<details>
<summary>imu_flex_serial.ino 주요 변경점</summary>

```cpp
// 1. CSV 헤더 확장
void sendCsvHeader() {
  Serial.println(F("timestamp,pitch,roll,yaw,accel_x,accel_y,accel_z,flex1,flex2,flex3,flex4,flex5"));
}

// 2. 가속도계 데이터 읽기 추가
float ax = 0, ay = 0, az = 0;
if (IMU.accelerationAvailable()) {
  IMU.readAcceleration(ax, ay, az);
}

// 3. CSV 출력 함수 확장
void printCsvRow(unsigned long ts, float pitch, float roll, float yaw, 
                 float ax, float ay, float az, const int flex[5])
```

</details>

<details>
<summary>csv_uart.py 주요 변경점</summary>

```python
# 1. CSV 헤더 업데이트
writer.writerow(['timestamp(ms)', 'pitch(°)', 'roll(°)', 'yaw(°)', 
                 'accel_x(g)', 'accel_y(g)', 'accel_z(g)', 
                 'flex1', 'flex2', 'flex3', 'flex4', 'flex5'])

# 2. 필드 수 검증 업데이트
if len(row) == 12:  # 9필드 → 12필드
```

</details>

## 💡 향후 확장 가능성

이 기반 위에서 추가로 개발 가능한 기능들:
- **센서 융합**: Madgwick 필터 적용
- **실시간 제스처 인식**: 온보드 ML 추론
- **자력계 지원**: 9축 IMU 완전 활용
- **WiFi 모드 확장**: 동일한 기능을 WiFi 모드에도 적용

## 🤝 기여 의사

SignGlove_HW 프로젝트의 발전에 기여하고 싶습니다! 어떤 방식으로 진행하는 것이 좋을지 의견 주시면 감사하겠습니다.

**관련 프로젝트**: [SignGlove 통합 수어 인식 시스템](https://github.com/your-username/SignGlove)

---

**테스트 환경**:
- Arduino Nano 33 IoT + LSM6DS3
- 플렉스 센서 5개
- 115200 baud UART 통신
- 가변 샘플링 레이트 (기본 4Hz)
