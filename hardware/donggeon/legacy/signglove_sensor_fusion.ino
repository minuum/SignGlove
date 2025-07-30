/*
  SignGlove Arduino 센서 퓨전 및 WiFi 통신 코드
  
  하드웨어: Arduino Nano 33 IoT + LSM6DS3 + 플렉스 센서 5개
  센서 퓨전: Madgwick Filter 적용
  통신: WiFi를 통한 HTTP POST 전송
  
  작성자: 이민우 (based on 양동건님 분석)
  날짜: 2025.07.15
*/

#include <WiFiNINA.h>
#include <Arduino_LSM6DS3.h>
#include <MadgwickAHRS.h>
#include <ArduinoJson.h>

// === WiFi 설정 ===
const char* ssid = "SignGlove_Network";       // WiFi SSID (환경변수에서 설정)
const char* password = "your_wifi_password";   // WiFi 비밀번호
const char* server_host = "192.168.1.100";    // 서버 IP (우분투 서버)
const int server_port = 8000;                 // 서버 포트

// === 하드웨어 핀 설정 ===
const int FLEX_PINS[5] = {A0, A1, A2, A3, A4};  // 플렉스 센서 핀
const int LED_PIN = LED_BUILTIN;
const int BUTTON_PIN = 2;                        // 제스처 시작/종료 버튼

// === 센서 설정 ===
const int SENSOR_SAMPLE_RATE = 104;             // LSM6DS3 기본 샘플링 레이트
const float ACCEL_SENSITIVITY = 4.0;            // ±4g 범위
const float GYRO_SENSITIVITY = 2000.0;          // ±2000 dps 범위
const int CALIBRATION_SAMPLES = 100;            // 캘리브레이션 샘플 수

// === 센서 퓨전 ===
Madgwick filter;                                 // Madgwick 필터 인스턴스
float sample_freq = SENSOR_SAMPLE_RATE;         // 샘플링 주파수

// === 센서 데이터 구조체 ===
struct SensorData {
  // 플렉스 센서 (0-1023)
  int flex_sensors[5];
  
  // 원시 IMU 데이터
  float accel_x, accel_y, accel_z;              // 가속도 (g)
  float gyro_x, gyro_y, gyro_z;                 // 각속도 (dps)
  
  // 센서 퓨전 결과
  float roll, pitch, yaw;                       // 오일러 각 (도)
  float q0, q1, q2, q3;                         // 쿼터니언
  
  // 방향 및 신뢰도
  String direction;                             // 방향 판단 결과
  float confidence;                             // 신뢰도 (0.0-1.0)
  
  // 메타데이터
  unsigned long timestamp;                      // 타임스탬프
  float battery_voltage;                        // 배터리 전압
  int wifi_rssi;                               // WiFi 신호 강도
};

// === 전역 변수 ===
SensorData current_data;
bool gesture_recording = false;
unsigned long gesture_start_time = 0;
float gyro_offset[3] = {0, 0, 0};              // 자이로스코프 오프셋 (캘리브레이션)
float accel_offset[3] = {0, 0, 0};             // 가속도계 오프셋

// === WiFi 클라이언트 ===
WiFiClient client;

void setup() {
  Serial.begin(9600);
  
  // 하드웨어 초기화
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  // 플렉스 센서 핀 설정
  for (int i = 0; i < 5; i++) {
    pinMode(FLEX_PINS[i], INPUT);
  }
  
  Serial.println("SignGlove Arduino 시작...");
  
  // IMU 센서 초기화
  if (!IMU.begin()) {
    Serial.println("❌ LSM6DS3 센서 초기화 실패!");
    while (1) {
      digitalWrite(LED_PIN, HIGH);
      delay(100);
      digitalWrite(LED_PIN, LOW);
      delay(100);
    }
  }
  
  Serial.println("✅ LSM6DS3 센서 초기화 완료");
  Serial.print("가속도계 샘플 레이트: ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("자이로스코프 샘플 레이트: ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");
  
  // Madgwick 필터 초기화
  filter.begin(sample_freq);
  Serial.println("✅ Madgwick 필터 초기화 완료");
  
  // 센서 캘리브레이션
  calibrateSensors();
  
  // WiFi 연결
  connectWiFi();
  
  Serial.println("🚀 SignGlove 준비 완료!");
  digitalWrite(LED_PIN, HIGH);  // 준비 완료 표시
}

void loop() {
  // 1. 센서 데이터 읽기
  readSensorData();
  
  // 2. 센서 퓨전 수행
  performSensorFusion();
  
  // 3. 방향 및 신뢰도 계산
  calculateOrientationAndConfidence();
  
  // 4. 제스처 버튼 확인
  checkGestureButton();
  
  // 5. 데이터 전송 (실시간 센서 데이터)
  if (millis() % (1000 / SENSOR_SAMPLE_RATE) == 0) {  // 104Hz로 전송
    sendSensorData();
  }
  
  // 6. 제스처 데이터 전송 (버튼으로 제어)
  if (gesture_recording) {
    // 제스처 진행중 표시
    digitalWrite(LED_PIN, millis() % 500 < 250);  // 깜빡임
  }
  
  delay(1);  // 1ms 딜레이
}

void readSensorData() {
  // 플렉스 센서 읽기
  for (int i = 0; i < 5; i++) {
    current_data.flex_sensors[i] = analogRead(FLEX_PINS[i]);
  }
  
  // IMU 센서 읽기
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(current_data.accel_x, current_data.accel_y, current_data.accel_z);
    
    // 캘리브레이션 오프셋 적용
    current_data.accel_x -= accel_offset[0];
    current_data.accel_y -= accel_offset[1];
    current_data.accel_z -= accel_offset[2];
  }
  
  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(current_data.gyro_x, current_data.gyro_y, current_data.gyro_z);
    
    // 캘리브레이션 오프셋 적용
    current_data.gyro_x -= gyro_offset[0];
    current_data.gyro_y -= gyro_offset[1];
    current_data.gyro_z -= gyro_offset[2];
  }
  
  // 메타데이터
  current_data.timestamp = millis();
  current_data.battery_voltage = readBatteryVoltage();
  current_data.wifi_rssi = WiFi.RSSI();
}

void performSensorFusion() {
  // Madgwick 필터 업데이트 (가속도 + 자이로)
  filter.updateIMU(
    current_data.gyro_x * PI / 180.0,    // dps를 rad/s로 변환
    current_data.gyro_y * PI / 180.0,
    current_data.gyro_z * PI / 180.0,
    current_data.accel_x,
    current_data.accel_y,
    current_data.accel_z
  );
  
  // 쿼터니언 가져오기
  current_data.q0 = filter.getQuaternionW();
  current_data.q1 = filter.getQuaternionX();
  current_data.q2 = filter.getQuaternionY();
  current_data.q3 = filter.getQuaternionZ();
  
  // 오일러 각 계산 (Gimbal Lock 주의하여 사용)
  current_data.roll = filter.getRoll();
  current_data.pitch = filter.getPitch();
  current_data.yaw = filter.getYaw();
}

void calculateOrientationAndConfidence() {
  // === 양동건님 개선 알고리즘 + 논문 기반 방향 판단 ===
  
  const float THRESHOLD = 0.6;  // 양동건님 설정 임계값
  const int NUM_SAMPLES = 10;   // 평균 필터 샘플 수
  
  // 1. 가속도 벡터 크기 계산
  float accel_magnitude = sqrt(
    current_data.accel_x * current_data.accel_x +
    current_data.accel_y * current_data.accel_y +
    current_data.accel_z * current_data.accel_z
  );
  
  // 2. 정규화된 가속도 벡터
  float norm_ax = current_data.accel_x / accel_magnitude;
  float norm_ay = current_data.accel_y / accel_magnitude;
  float norm_az = current_data.accel_z / accel_magnitude;
  
  // 3. 중력과의 차이 계산 (신뢰도 결정)
  float gravity_diff = abs(accel_magnitude - 1.0);  // 1g와의 차이
  current_data.confidence = max(0.0, 1.0 - gravity_diff);
  
  // 4. 움직임 감지
  float motion_intensity = abs(current_data.gyro_x) + 
                          abs(current_data.gyro_y) + 
                          abs(current_data.gyro_z);
  bool is_moving = motion_intensity > 30.0;  // 30 dps 임계값
  
  // 5. 양동건님 개선된 방향 판단 (6축 활용)
  if (is_moving) {
    // 움직임 중: 자이로스코프 기반 판단
    if (abs(current_data.gyro_z) > 50) {
      current_data.direction = (current_data.gyro_z > 0) ? "시계방향 회전" : "반시계방향 회전";
      current_data.confidence *= 0.8;  // 움직임 중 신뢰도 감소
    } else {
      current_data.direction = "움직임 중";
      current_data.confidence *= 0.5;
    }
  } else {
    // 정적 상태: 개선된 6축 방향 판단
    if (norm_az < -THRESHOLD) {
      current_data.direction = "손바닥이 위 (하늘)";
    } else if (norm_az > THRESHOLD) {
      current_data.direction = "손바닥이 아래 (땅)";
    } else if (norm_ay < -THRESHOLD) {
      current_data.direction = "손바닥이 앞 (사람 방향)";
    } else if (norm_ay > THRESHOLD) {
      current_data.direction = "손바닥이 뒤 (자기 쪽)";
    } else if (norm_ax < -THRESHOLD) {
      current_data.direction = "손이 오른쪽으로 기울어짐";
    } else if (norm_ax > THRESHOLD) {
      current_data.direction = "손이 왼쪽으로 기울어짐";
    } else {
      current_data.direction = "방향 모호함";  // 양동건님 케이스 추가
      current_data.confidence *= 0.4;  // 모호함 시 신뢰도 대폭 감소
    }
  }
  
  // 6. 신뢰도 최종 조정 (WiFi 신호, 배터리 상태 고려)
  if (current_data.wifi_rssi < -70) {
    current_data.confidence *= 0.95;  // 약한 신호
  }
  if (current_data.battery_voltage < 3.3) {
    current_data.confidence *= 0.9;   // 배터리 부족
  }
}

void checkGestureButton() {
  static bool last_button_state = HIGH;
  bool current_button_state = digitalRead(BUTTON_PIN);
  
  // 버튼 눌림 감지 (falling edge)
  if (last_button_state == HIGH && current_button_state == LOW) {
    if (!gesture_recording) {
      // 제스처 시작
      gesture_recording = true;
      gesture_start_time = millis();
      Serial.println("🎯 제스처 녹화 시작");
    } else {
      // 제스처 종료
      gesture_recording = false;
      unsigned long duration = millis() - gesture_start_time;
      Serial.print("⏹️ 제스처 녹화 종료 (");
      Serial.print(duration);
      Serial.println("ms)");
      
      // 제스처 데이터 전송
      sendGestureData(duration);
      
      digitalWrite(LED_PIN, HIGH);  // 완료 표시
    }
    delay(50);  // 디바운싱
  }
  
  last_button_state = current_button_state;
}

void sendSensorData() {
  if (WiFi.status() != WL_CONNECTED) {
    reconnectWiFi();
    return;
  }
  
  // JSON 데이터 생성
  DynamicJsonDocument doc(1024);
  doc["device_id"] = "ARDUINO_SIGNGLOVE_001";
  doc["timestamp"] = current_data.timestamp;
  
  // 플렉스 센서 배열
  JsonArray flex_array = doc.createNestedArray("flex_sensors");
  for (int i = 0; i < 5; i++) {
    flex_array.add(current_data.flex_sensors[i]);
  }
  
  // IMU 센서 데이터
  JsonObject gyro = doc.createNestedObject("gyroscope");
  gyro["accel_x"] = current_data.accel_x;
  gyro["accel_y"] = current_data.accel_y;
  gyro["accel_z"] = current_data.accel_z;
  gyro["gyro_x"] = current_data.gyro_x;
  gyro["gyro_y"] = current_data.gyro_y;
  gyro["gyro_z"] = current_data.gyro_z;
  gyro["roll"] = current_data.roll;
  gyro["pitch"] = current_data.pitch;
  gyro["yaw"] = current_data.yaw;
  gyro["q0"] = current_data.q0;
  gyro["q1"] = current_data.q1;
  gyro["q2"] = current_data.q2;
  gyro["q3"] = current_data.q3;
  gyro["direction"] = current_data.direction;
  
  doc["quality_score"] = current_data.confidence;
  doc["battery_voltage"] = current_data.battery_voltage;
  doc["wifi_rssi"] = current_data.wifi_rssi;
  
  // HTTP POST 요청
  if (client.connect(server_host, server_port)) {
    client.println("POST /data/sensor HTTP/1.1");
    client.print("Host: ");
    client.println(server_host);
    client.println("Content-Type: application/json");
    client.print("Content-Length: ");
    client.println(measureJson(doc));
    client.println();
    serializeJson(doc, client);
    client.println();
    client.stop();
  }
}

void sendGestureData(unsigned long duration) {
  if (WiFi.status() != WL_CONNECTED) return;
  
  DynamicJsonDocument doc(512);
  doc["device_id"] = "ARDUINO_SIGNGLOVE_001";
  doc["timestamp"] = millis();
  doc["gesture_class"] = "미정의_제스처";  // TODO: 실제 제스처 인식
  doc["confidence"] = current_data.confidence;
  doc["duration_ms"] = duration;
  doc["session_id"] = String(gesture_start_time);
  
  // 센서 스냅샷
  JsonObject snapshot = doc.createNestedObject("sensor_snapshot");
  JsonArray flex_snap = snapshot.createNestedArray("flex_sensors");
  for (int i = 0; i < 5; i++) {
    flex_snap.add(current_data.flex_sensors[i]);
  }
  snapshot["direction"] = current_data.direction;
  snapshot["roll"] = current_data.roll;
  snapshot["pitch"] = current_data.pitch;
  snapshot["yaw"] = current_data.yaw;
  
  // HTTP POST 요청
  if (client.connect(server_host, server_port)) {
    client.println("POST /data/gesture HTTP/1.1");
    client.print("Host: ");
    client.println(server_host);
    client.println("Content-Type: application/json");
    client.print("Content-Length: ");
    client.println(measureJson(doc));
    client.println();
    serializeJson(doc, client);
    client.println();
    client.stop();
  }
}

void connectWiFi() {
  Serial.print("WiFi 연결 중... ");
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.print("✅ WiFi 연결 성공: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("❌ WiFi 연결 실패");
  }
}

void reconnectWiFi() {
  static unsigned long last_attempt = 0;
  
  if (millis() - last_attempt > 10000) {  // 10초마다 재시도
    Serial.println("WiFi 재연결 시도...");
    connectWiFi();
    last_attempt = millis();
  }
}

void calibrateSensors() {
  Serial.println("센서 캘리브레이션 시작... (5초간 가만히 두세요)");
  
  float accel_sum[3] = {0, 0, 0};
  float gyro_sum[3] = {0, 0, 0};
  
  for (int i = 0; i < CALIBRATION_SAMPLES; i++) {
    if (IMU.accelerationAvailable()) {
      float ax, ay, az;
      IMU.readAcceleration(ax, ay, az);
      accel_sum[0] += ax;
      accel_sum[1] += ay;
      accel_sum[2] += az - 1.0;  // 중력 보정
    }
    
    if (IMU.gyroscopeAvailable()) {
      float gx, gy, gz;
      IMU.readGyroscope(gx, gy, gz);
      gyro_sum[0] += gx;
      gyro_sum[1] += gy;
      gyro_sum[2] += gz;
    }
    
    delay(50);
    if (i % 20 == 0) Serial.print(".");
  }
  
  // 오프셋 계산
  for (int i = 0; i < 3; i++) {
    accel_offset[i] = accel_sum[i] / CALIBRATION_SAMPLES;
    gyro_offset[i] = gyro_sum[i] / CALIBRATION_SAMPLES;
  }
  
  Serial.println();
  Serial.println("✅ 캘리브레이션 완료");
  Serial.print("자이로 오프셋: ");
  Serial.print(gyro_offset[0]); Serial.print(", ");
  Serial.print(gyro_offset[1]); Serial.print(", ");
  Serial.println(gyro_offset[2]);
}

float readBatteryVoltage() {
  // Arduino Nano 33 IoT는 내장 배터리 모니터링 없음
  // 외부 전압 분배기 회로 필요
  // 임시로 고정값 반환
  return 4.2;  // 임시값
} 