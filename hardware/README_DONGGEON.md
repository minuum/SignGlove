# 양동건 팀원 작업 가이드 📋

## 🎯 **당신의 역할**
- **아두이노 펌웨어**: 센서 데이터 읽기 및 전송
- **UART/WiFi 통신**: 아두이노 ↔ 노트북 통신 구현
- **클라이언트 프로그램**: 노트북 → 서버 HTTP 요청

## 📁 **작업 디렉토리 구조**
```
hardware/
├── arduino/
│   ├── signglove_firmware.ino        # 메인 펌웨어 (당신이 작성)
│   ├── sensor_reader.h               # 센서 읽기 모듈
│   ├── wifi_module.h                 # WiFi 통신 모듈
│   └── uart_module.h                 # UART 통신 모듈
├── client/
│   ├── data_client.py                # 노트북 클라이언트 (당신이 작성)
│   └── communication_manager.py      # 통신 관리자
└── test/
    ├── sensor_test.ino               # 센서 테스트
    └── communication_test.py         # 통신 테스트
```

## 🔧 **1단계: 아두이노 펌웨어**

### **요구사항**
- 5개 플렉스 센서 값 읽기 (아날로그 입력)
- 6DOF IMU 센서 값 읽기 (I2C 통신)
- 20Hz 샘플링 레이트 유지
- JSON 또는 CSV 형식으로 데이터 출력
- UART/WiFi 중 선택 가능

### **데이터 포맷**
```json
{
    "device_id": "ARDUINO_001",
    "timestamp": "2025-07-26T21:30:00.000Z",
    "flex_sensors": {
        "flex_1": 850.5,
        "flex_2": 300.2,
        "flex_3": 285.7,
        "flex_4": 310.1,
        "flex_5": 295.8
    },
    "gyro_data": {
        "gyro_x": 2.15,
        "gyro_y": -1.05,
        "gyro_z": 0.32,
        "accel_x": 0.21,
        "accel_y": -9.78,
        "accel_z": 0.15
    },
    "battery_level": 95.2,
    "signal_strength": -45
}
```

### **CSV 포맷 (대안)**
```csv
timestamp,device_id,flex_1,flex_2,flex_3,flex_4,flex_5,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z,battery_level,signal_strength
2025-07-26T21:30:00.000Z,ARDUINO_001,850.5,300.2,285.7,310.1,295.8,2.15,-1.05,0.32,0.21,-9.78,0.15,95.2,-45
```

## 🔧 **2단계: 노트북 클라이언트**

### **요구사항**
- 아두이노에서 UART/WiFi로 데이터 수신
- 서버 API와 통신
- 실험 세션 관리
- 실시간 모니터링

### **서버 API 엔드포인트**
```
서버 주소: http://localhost:8000

1. 실험 시작: POST /experiment/start
2. 센서 데이터 전송: POST /data/sensor  
3. 샘플 완료: POST /sample/complete
4. 실험 종료: POST /experiment/stop
5. 상태 확인: GET /status
```

### **클라이언트 워크플로우**
1. 서버에 실험 세션 시작 요청
2. 아두이노에서 센서 데이터 수신
3. 수신한 데이터를 서버로 전송
4. 샘플 완료 시 서버에 알림
5. 실험 종료 시 세션 종료

## 🚀 **3단계: 테스트 및 검증**

### **단위 테스트**
- 센서 값 정확성 검증
- 통신 안정성 테스트
- 샘플링 레이트 측정

### **통합 테스트**
- 아두이노 ↔ 노트북 ↔ 서버 전체 파이프라인
- 실제 데이터 수집 테스트

## 📞 **이민우와의 인터페이스**

### **당신이 보내야 할 데이터**
```python
# HTTP POST 요청 예시
import requests

# 1. 실험 시작
response = requests.post("http://localhost:8000/experiment/start", json={
    "session_id": "session_20250730_100000",
    "performer_id": "donggeon",
    "class_label": "ㄱ",
    "category": "consonant",
    "target_samples": 60,
    "duration_per_sample": 5,
    "sampling_rate": 20
})

# 2. 센서 데이터 전송
sensor_data = {
    "device_id": "ARDUINO_001",
    "timestamp": "2025-07-30T10:00:00.000Z",
    "flex_sensors": {
        "flex_1": 850.5,
        "flex_2": 300.2,
        "flex_3": 285.7,
        "flex_4": 310.1,
        "flex_5": 295.8
    },
    "gyro_data": {
        "gyro_x": 2.15,
        "gyro_y": -1.05,
        "gyro_z": 0.32,
        "accel_x": 0.21,
        "accel_y": -9.78,
        "accel_z": 0.15
    },
    "battery_level": 95.2,
    "signal_strength": -45
}

response = requests.post("http://localhost:8000/data/sensor", json=sensor_data)
```

## ⚡ **우선순위 작업**

### **Week 1 (7/26 - 7/30)**
1. ✅ **아두이노 펌웨어 기본 구조**
   - 센서 읽기 테스트
   - UART 통신 구현
   - 20Hz 샘플링 구현

2. ✅ **노트북 클라이언트 기본**
   - 시리얼 통신 수신
   - HTTP 요청 전송
   - 기본 워크플로우

3. ✅ **통합 테스트**
   - 아두이노-노트북-서버 연결
   - 실제 데이터 수집 검증

### **Meeting 5 목표 (7/30)**
- 완전한 파이프라인 동작
- 최소 1개 클래스 데이터 수집 성공
- 시스템 안정성 검증

## 📝 **개발 가이드라인**

### **코딩 컨벤션**
- 함수명과 변수명: 영어로, snake_case
- 주석: 한국어로 작성
- 에러 처리 필수 포함

### **통신 프로토콜**
- **UART**: 115200 baud rate
- **WiFi**: HTTP/JSON 프로토콜
- **타임아웃**: 5초 이내 응답

### **품질 관리**
- 센서 값 범위 검증
- 통신 오류 재시도 (최대 3회)
- 실시간 로깅

## 🆘 **문제 해결**

### **일반적인 문제들**
1. **센서 값 불안정**: 하드웨어 연결 재확인
2. **통신 끊김**: 타임아웃 설정 조정
3. **샘플링 레이트 부정확**: 타이머 인터럽트 사용

### **도움이 필요할 때**
- 이민우에게 연락: 서버 API 관련
- 팀 미팅에서 논의: 전체 시스템 이슈

## 🎯 **최종 목표**

**Meeting 5 (7/30)에서 데모할 수 있는 완전한 시스템:**
1. 아두이노에서 안정적인 센서 데이터 읽기
2. 노트북으로 실시간 데이터 전송
3. 서버에서 베스트 프랙티스 구조로 저장
4. 웹 대시보드에서 실시간 모니터링

**성공 기준:**
- 5초 동안 안정적인 20Hz 데이터 수집
- 데이터 손실률 < 5%
- 전체 파이프라인 지연시간 < 100ms

화이팅! 💪 