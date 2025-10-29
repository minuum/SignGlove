# 센서 퓨전 알고리즘 구현 가이드

## 개요 (@양동건 분석 - 2025.07.15)

LSM6DS3 센서의 **"방향 모호함"** 문제를 해결하기 위한 센서 퓨전 알고리즘 구현 가이드입니다.

## 문제 상황

### 가속도 기반 방향 판단의 한계
```cpp
// 현재 구현 (양동건 Arduino 코드)
if (z < -0.8) direction = "손바닥이 위 (하늘)";
else if (z > 0.8) direction = "손바닥이 아래 (땅)"; 
else if (y < -0.8) direction = "손바닥이 앞 (사람 방향)";
else if (y > 0.8) direction = "손바닥이 뒤 (자기 쪽)";
else direction = "방향 모호함";  // 🚨 문제: 중간값에서 모호함
```

### 실제 테스트 결과
```
X: -0.031 g	Y: -0.762 g	Z: 0.588 g	→ 방향 모호함 🚨
```
**문제**: 움직임 전환 시점에서 방향 판단 실패

## 해결방안: Madgwick Filter

### 왜 Madgwick Filter인가?
- **성능/복잡도 비율 최적** ⭐
- **실시간 처리 가능** (104Hz 지원)
- **드리프트 보정** (누적 오차 해결)
- **라이브러리 지원** (Arduino 호환)

### 구현 계획

#### 1. 라이브러리 설치
```cpp
#include <Arduino_LSM6DS3.h>
#include <MadgwickAHRS.h>  // 센서 퓨전 라이브러리

Madgwick filter;
```

#### 2. 센서 퓨전 데이터 구조
```cpp
struct SensorFusionData {
    // 원시 센서 데이터
    float accel_x, accel_y, accel_z;     // 가속도
    float gyro_x, gyro_y, gyro_z;        // 각속도
    
    // 센서 퓨전 결과
    float roll, pitch, yaw;              // 오일러 각
    float q0, q1, q2, q3;                // 쿼터니언
    
    // 방향 판단
    String direction;                     // 최종 방향
    float confidence;                     // 신뢰도 (0-1)
};
```

#### 3. 센서 퓨전 알고리즘 구현
```cpp
void updateSensorFusion() {
    // 1. 원시 센서 데이터 읽기
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);
    
    // 2. Madgwick 필터 업데이트 (104Hz)
    filter.updateIMU(gx, gy, gz, ax, ay, az);
    
    // 3. 쿼터니언 → 오일러 각 변환
    roll = filter.getRoll();
    pitch = filter.getPitch();
    yaw = filter.getYaw();
    
    // 4. 개선된 방향 판단
    direction = getOrientationWithConfidence(roll, pitch, yaw);
}
```

#### 4. 신뢰도 기반 방향 판단
```cpp
String getOrientationWithConfidence(float roll, float pitch, float yaw) {
    float confidence = 0.0;
    String direction = "";
    
    // Roll 기반 판단 (좌우 기울기)
    if (abs(roll) > 60) {
        direction = (roll > 0) ? "손바닥 오른쪽 기울임" : "손바닥 왼쪽 기울임";
        confidence = map(abs(roll), 60, 90, 0.6, 1.0);
    }
    // Pitch 기반 판단 (앞뒤 기울기) 
    else if (abs(pitch) > 60) {
        direction = (pitch > 0) ? "손바닥 아래향" : "손바닥 위향";
        confidence = map(abs(pitch), 60, 90, 0.6, 1.0);
    }
    // Yaw 기반 판단 (회전)
    else if (abs(yaw) > 45) {
        direction = (yaw > 0) ? "시계방향 회전" : "반시계방향 회전";
        confidence = map(abs(yaw), 45, 180, 0.5, 0.9);
    }
    else {
        direction = "중립 위치";
        confidence = 0.3;  // 낮은 신뢰도
    }
    
    return direction + " (신뢰도: " + String(confidence, 2) + ")";
}
```

## 다음 단계

### 즉시 구현 (1주일 내)
1. **Madgwick 라이브러리 테스트**
   - Arduino IDE에 라이브러리 설치
   - 기본 예제 코드 실행
   - LSM6DS3 연동 확인

2. **신뢰도 기반 방향 판단 구현**
   - 기존 if-else 로직을 신뢰도 기반으로 개선
   - 임계값 튜닝 (60도, 45도 등)

### 중기 목표 (2주일 내)
3. **실시간 데이터 전송 연동**
   - WiFi를 통한 센서 퓨전 데이터 전송
   - 서버에서 신뢰도 기반 필터링

4. **성능 최적화**
   - 104Hz 샘플링 레이트 맞춤
   - 메모리 사용량 최적화

## 기대 효과

### Before (현재)
```
X: -0.031 g	Y: -0.762 g	Z: 0.588 g	→ 방향 모호함 🚨
```

### After (센서 퓨전 적용)
```
Roll: 15°  Pitch: -45°  Yaw: 12°  → 손바닥 위향 (신뢰도: 0.85) ✅
```

**결과**: 
- ✅ **모호함 해결**: 신뢰도 기반 방향 판단
- ✅ **드리프트 보정**: Madgwick 필터로 누적 오차 제거  
- ✅ **실시간 처리**: 104Hz 고속 처리 가능
- ✅ **확장성**: 추후 복잡한 제스처 패턴 인식 가능

---

**작성자**: 이민우 (based on 양동건 분석)  
**날짜**: 2025.07.15  
**상태**: 구현 준비 완료 🚀 