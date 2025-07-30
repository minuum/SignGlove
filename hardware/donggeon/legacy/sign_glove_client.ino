/*
 * SignGlove Arduino Client
 * 
 * 플렉스 센서 5개 + 자이로 센서(6DOF) 데이터를 수집하여
 * WiFi를 통해 FastAPI 서버로 전송하는 아두이노 코드
 * 
 * 하드웨어 구성:
 * - 아두이노 보드 (ESP32 권장)
 * - 플렉스 센서 5개 (A0~A4 핀)
 * - 자이로 센서 MPU6050 (I2C 통신)
 * - WiFi 모듈 (ESP32 내장)
 * 
 * 작성자: 이민우
 * 최종 수정: 2024-01-03
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <MPU6050.h>

// WiFi 설정
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// 서버 설정
const char* server_url = "http://YOUR_SERVER_IP:8000/data/sensor";
const String device_id = "SIGNGLOVE_001";

// 센서 핀 설정
const int flex_pins[5] = {A0, A1, A2, A3, A4};  // 플렉스 센서 핀
const int battery_pin = A5;  // 배터리 모니터링 핀

// 자이로 센서 객체
MPU6050 mpu;

// 데이터 수집 간격 (밀리초)
const unsigned long data_interval = 50;  // 20Hz 수집률
unsigned long last_data_time = 0;

// 연결 상태 확인
bool wifi_connected = false;
bool server_connected = false;
unsigned long last_connection_check = 0;
const unsigned long connection_check_interval = 5000;  // 5초마다 확인

// 데이터 구조체
struct SensorData {
  float flex_sensors[5];
  float gyro_x, gyro_y, gyro_z;
  float accel_x, accel_y, accel_z;
  float battery_level;
  int signal_strength;
  unsigned long timestamp;
};

void setup() {
  Serial.begin(115200);
  Serial.println("SignGlove 시스템 시작");
  
  // 센서 초기화
  init_sensors();
  
  // WiFi 연결
  connect_wifi();
  
  // 자이로 센서 초기화
  init_gyro();
  
  Serial.println("초기화 완료");
}

void loop() {
  unsigned long current_time = millis();
  
  // 연결 상태 확인
  if (current_time - last_connection_check > connection_check_interval) {
    check_connections();
    last_connection_check = current_time;
  }
  
  // 데이터 수집 및 전송
  if (current_time - last_data_time > data_interval) {
    if (wifi_connected) {
      SensorData data;
      collect_sensor_data(data);
      send_data_to_server(data);
    }
    last_data_time = current_time;
  }
  
  // 시리얼 모니터 명령 처리
  handle_serial_commands();
  
  delay(10);  // 짧은 지연
}

void init_sensors() {
  // 플렉스 센서 핀 초기화
  for (int i = 0; i < 5; i++) {
    pinMode(flex_pins[i], INPUT);
  }
  
  // 배터리 모니터링 핀 초기화
  pinMode(battery_pin, INPUT);
  
  Serial.println("센서 초기화 완료");
}

void connect_wifi() {
  WiFi.begin(ssid, password);
  Serial.print("WiFi 연결 중");
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    wifi_connected = true;
    Serial.println();
    Serial.print("WiFi 연결 성공: ");
    Serial.println(WiFi.localIP());
  } else {
    wifi_connected = false;
    Serial.println();
    Serial.println("WiFi 연결 실패");
  }
}

void init_gyro() {
  Wire.begin();
  mpu.initialize();
  
  if (mpu.testConnection()) {
    Serial.println("MPU6050 연결 성공");
  } else {
    Serial.println("MPU6050 연결 실패");
  }
}

void collect_sensor_data(SensorData& data) {
  // 플렉스 센서 데이터 수집
  for (int i = 0; i < 5; i++) {
    data.flex_sensors[i] = analogRead(flex_pins[i]);
  }
  
  // 자이로 센서 데이터 수집
  int16_t ax, ay, az;
  int16_t gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  
  // 자이로 데이터 변환 (°/s)
  data.gyro_x = gx / 131.0;
  data.gyro_y = gy / 131.0;
  data.gyro_z = gz / 131.0;
  
  // 가속도 데이터 변환 (m/s²)
  data.accel_x = ax / 16384.0 * 9.81;
  data.accel_y = ay / 16384.0 * 9.81;
  data.accel_z = az / 16384.0 * 9.81;
  
  // 배터리 레벨 계산
  int battery_reading = analogRead(battery_pin);
  data.battery_level = map(battery_reading, 0, 1023, 0, 100);
  
  // WiFi 신호 강도
  data.signal_strength = WiFi.RSSI();
  
  // 타임스탬프
  data.timestamp = millis();
}

void send_data_to_server(SensorData& data) {
  if (!wifi_connected) return;
  
  HTTPClient http;
  http.begin(server_url);
  http.addHeader("Content-Type", "application/json");
  
  // JSON 데이터 생성
  StaticJsonDocument<1024> doc;
  doc["device_id"] = device_id;
  doc["timestamp"] = get_iso_timestamp();
  
  // 플렉스 센서 데이터
  JsonObject flex_sensors = doc.createNestedObject("flex_sensors");
  flex_sensors["flex_1"] = data.flex_sensors[0];
  flex_sensors["flex_2"] = data.flex_sensors[1];
  flex_sensors["flex_3"] = data.flex_sensors[2];
  flex_sensors["flex_4"] = data.flex_sensors[3];
  flex_sensors["flex_5"] = data.flex_sensors[4];
  
  // 자이로 센서 데이터
  JsonObject gyro_data = doc.createNestedObject("gyro_data");
  gyro_data["gyro_x"] = data.gyro_x;
  gyro_data["gyro_y"] = data.gyro_y;
  gyro_data["gyro_z"] = data.gyro_z;
  gyro_data["accel_x"] = data.accel_x;
  gyro_data["accel_y"] = data.accel_y;
  gyro_data["accel_z"] = data.accel_z;
  
  // 디바이스 상태
  doc["battery_level"] = data.battery_level;
  doc["signal_strength"] = data.signal_strength;
  
  // JSON 문자열 생성
  String json_string;
  serializeJson(doc, json_string);
  
  // HTTP POST 요청
  int http_response = http.POST(json_string);
  
  if (http_response > 0) {
    String response = http.getString();
    server_connected = true;
    
    // 성공 응답 처리
    if (http_response == 200) {
      Serial.println("데이터 전송 성공");
    } else {
      Serial.print("서버 응답 오류: ");
      Serial.println(http_response);
    }
  } else {
    server_connected = false;
    Serial.print("HTTP 요청 실패: ");
    Serial.println(http_response);
  }
  
  http.end();
}

void check_connections() {
  // WiFi 연결 확인
  if (WiFi.status() != WL_CONNECTED) {
    wifi_connected = false;
    Serial.println("WiFi 연결 끊어짐, 재연결 시도");
    connect_wifi();
  }
}

String get_iso_timestamp() {
  // 간단한 ISO 타임스탬프 생성 (실제로는 NTP 동기화 필요)
  unsigned long timestamp = millis();
  return String(timestamp);
}

void handle_serial_commands() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "status") {
      print_status();
    } else if (command == "test") {
      test_sensors();
    } else if (command == "reset") {
      ESP.restart();
    } else if (command == "help") {
      print_help();
    }
  }
}

void print_status() {
  Serial.println("=== SignGlove 상태 ===");
  Serial.print("디바이스 ID: ");
  Serial.println(device_id);
  Serial.print("WiFi 연결: ");
  Serial.println(wifi_connected ? "연결됨" : "연결 안됨");
  Serial.print("서버 연결: ");
  Serial.println(server_connected ? "연결됨" : "연결 안됨");
  Serial.print("IP 주소: ");
  Serial.println(WiFi.localIP());
  Serial.print("신호 강도: ");
  Serial.print(WiFi.RSSI());
  Serial.println(" dBm");
  Serial.print("업타임: ");
  Serial.print(millis() / 1000);
  Serial.println(" 초");
}

void test_sensors() {
  Serial.println("=== 센서 테스트 ===");
  
  // 플렉스 센서 테스트
  Serial.println("플렉스 센서 값:");
  for (int i = 0; i < 5; i++) {
    Serial.print("  Flex ");
    Serial.print(i + 1);
    Serial.print(": ");
    Serial.println(analogRead(flex_pins[i]));
  }
  
  // 자이로 센서 테스트
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  
  Serial.println("자이로 센서 값:");
  Serial.print("  Gyro X: ");
  Serial.println(gx / 131.0);
  Serial.print("  Gyro Y: ");
  Serial.println(gy / 131.0);
  Serial.print("  Gyro Z: ");
  Serial.println(gz / 131.0);
  Serial.print("  Accel X: ");
  Serial.println(ax / 16384.0);
  Serial.print("  Accel Y: ");
  Serial.println(ay / 16384.0);
  Serial.print("  Accel Z: ");
  Serial.println(az / 16384.0);
  
  // 배터리 테스트
  Serial.print("배터리 레벨: ");
  Serial.print(map(analogRead(battery_pin), 0, 1023, 0, 100));
  Serial.println("%");
}

void print_help() {
  Serial.println("=== 사용 가능한 명령어 ===");
  Serial.println("status - 시스템 상태 확인");
  Serial.println("test - 센서 테스트");
  Serial.println("reset - 시스템 재시작");
  Serial.println("help - 도움말 표시");
} 