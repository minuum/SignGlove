/*
 * SignGlove 펌웨어 템플릿
 * 
 * 작성자: 양동건
 * 목적: 플렉스 센서 및 IMU 센서 데이터 수집
 * 통신: UART/WiFi를 통해 노트북으로 전송
 */

#include <WiFi.h>
#include <ArduinoJson.h>
#include <Wire.h>
// #include <MPU6050.h>  // IMU 센서 라이브러리 (설치 필요)

// ===== 설정 상수 =====
#define SAMPLING_RATE_HZ 20          // 샘플링 주파수
#define SAMPLING_INTERVAL_MS (1000 / SAMPLING_RATE_HZ)

// 플렉스 센서 핀 (아날로그 입력)
#define FLEX_PIN_1 A0
#define FLEX_PIN_2 A1  
#define FLEX_PIN_3 A2
#define FLEX_PIN_4 A3
#define FLEX_PIN_5 A4

// IMU 센서 I2C 주소
#define MPU6050_ADDR 0x68

// WiFi 설정 (필요시 사용)
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// 디바이스 ID
String device_id = "ARDUINO_001";

// ===== 전역 변수 =====
unsigned long last_sample_time = 0;
bool wifi_enabled = false;  // false: UART 모드, true: WiFi 모드

// ===== 센서 데이터 구조체 =====
struct SensorData {
    String timestamp;
    String device_id;
    
    // 플렉스 센서 값
    float flex_1, flex_2, flex_3, flex_4, flex_5;
    
    // IMU 센서 값
    float gyro_x, gyro_y, gyro_z;
    float accel_x, accel_y, accel_z;
    
    // 시스템 정보
    float battery_level;
    int signal_strength;
};

void setup() {
    // 시리얼 통신 초기화
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("=== SignGlove 펌웨어 시작 ===");
    
    // 센서 초기화
    initialize_sensors();
    
    // 통신 방식 선택
    if (wifi_enabled) {
        initialize_wifi();
    } else {
        Serial.println("UART 모드로 시작");
    }
    
    Serial.println("초기화 완료. 데이터 수집 시작...");
}

void loop() {
    unsigned long current_time = millis();
    
    // 샘플링 타이밍 체크
    if (current_time - last_sample_time >= SAMPLING_INTERVAL_MS) {
        // 센서 데이터 읽기
        SensorData data = read_sensor_data();
        
        // 데이터 전송
        send_sensor_data(data);
        
        last_sample_time = current_time;
    }
    
    // 다른 작업들...
    delay(1); // 짧은 대기
}

void initialize_sensors() {
    Serial.println("센서 초기화 중...");
    
    // 플렉스 센서 핀 설정 (아날로그 입력은 별도 설정 불필요)
    
    // IMU 센서 초기화 (I2C)
    Wire.begin();
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x6B); // PWR_MGMT_1 레지스터
    Wire.write(0);    // wake up
    Wire.endTransmission(true);
    
    Serial.println("센서 초기화 완료");
}

void initialize_wifi() {
    Serial.println("WiFi 연결 중...");
    
    WiFi.begin(ssid, password);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nWiFi 연결 성공!");
        Serial.print("IP 주소: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("\nWiFi 연결 실패. UART 모드로 전환.");
        wifi_enabled = false;
    }
}

SensorData read_sensor_data() {
    SensorData data;
    
    // 타임스탬프 생성
    data.timestamp = get_timestamp();
    data.device_id = device_id;
    
    // 플렉스 센서 읽기 (0-1023 아날로그 값)
    data.flex_1 = analogRead(FLEX_PIN_1);
    data.flex_2 = analogRead(FLEX_PIN_2);
    data.flex_3 = analogRead(FLEX_PIN_3);
    data.flex_4 = analogRead(FLEX_PIN_4);
    data.flex_5 = analogRead(FLEX_PIN_5);
    
    // IMU 센서 읽기
    read_imu_data(data);
    
    // 시스템 정보
    data.battery_level = read_battery_level();
    data.signal_strength = get_signal_strength();
    
    return data;
}

void read_imu_data(SensorData& data) {
    // MPU6050에서 자이로 및 가속도 데이터 읽기
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x3B); // ACCEL_XOUT_H 레지스터
    Wire.endTransmission(false);
    Wire.requestFrom(MPU6050_ADDR, 14, true);
    
    // 가속도 값 읽기 (16비트)
    int16_t accel_x_raw = Wire.read() << 8 | Wire.read();
    int16_t accel_y_raw = Wire.read() << 8 | Wire.read();
    int16_t accel_z_raw = Wire.read() << 8 | Wire.read();
    
    // 온도 값 건너뛰기
    Wire.read(); Wire.read();
    
    // 자이로 값 읽기 (16비트)
    int16_t gyro_x_raw = Wire.read() << 8 | Wire.read();
    int16_t gyro_y_raw = Wire.read() << 8 | Wire.read();
    int16_t gyro_z_raw = Wire.read() << 8 | Wire.read();
    
    // 스케일 변환 (MPU6050 스펙에 따라 조정)
    data.accel_x = accel_x_raw / 16384.0; // ±2g 설정 시
    data.accel_y = accel_y_raw / 16384.0;
    data.accel_z = accel_z_raw / 16384.0;
    
    data.gyro_x = gyro_x_raw / 131.0; // ±250°/s 설정 시
    data.gyro_y = gyro_y_raw / 131.0;
    data.gyro_z = gyro_z_raw / 131.0;
}

String get_timestamp() {
    // 간단한 타임스탬프 (실제로는 RTC 또는 NTP 사용 권장)
    unsigned long current_millis = millis();
    return String(current_millis);
}

float read_battery_level() {
    // 배터리 레벨 읽기 (예시)
    // 실제 구현은 하드웨어에 따라 다름
    return 95.0; // 임시값
}

int get_signal_strength() {
    if (wifi_enabled && WiFi.status() == WL_CONNECTED) {
        return WiFi.RSSI();
    }
    return -50; // 기본값
}

void send_sensor_data(const SensorData& data) {
    if (wifi_enabled) {
        send_via_wifi(data);
    } else {
        send_via_uart(data);
    }
}

void send_via_uart(const SensorData& data) {
    // JSON 형식으로 시리얼 출력
    DynamicJsonDocument doc(1024);
    
    doc["device_id"] = data.device_id;
    doc["timestamp"] = data.timestamp;
    
    JsonObject flex_sensors = doc.createNestedObject("flex_sensors");
    flex_sensors["flex_1"] = data.flex_1;
    flex_sensors["flex_2"] = data.flex_2;
    flex_sensors["flex_3"] = data.flex_3;
    flex_sensors["flex_4"] = data.flex_4;
    flex_sensors["flex_5"] = data.flex_5;
    
    JsonObject gyro_data = doc.createNestedObject("gyro_data");
    gyro_data["gyro_x"] = data.gyro_x;
    gyro_data["gyro_y"] = data.gyro_y;
    gyro_data["gyro_z"] = data.gyro_z;
    gyro_data["accel_x"] = data.accel_x;
    gyro_data["accel_y"] = data.accel_y;
    gyro_data["accel_z"] = data.accel_z;
    
    doc["battery_level"] = data.battery_level;
    doc["signal_strength"] = data.signal_strength;
    
    // JSON 문자열로 출력
    serializeJson(doc, Serial);
    Serial.println(); // 줄바꿈
}

void send_via_wifi(const SensorData& data) {
    // WiFi를 통한 HTTP 전송 (구현 필요)
    // 노트북의 클라이언트 프로그램이 받을 수 있도록
    Serial.println("WiFi 전송 모드 - 구현 필요");
}

// ===== 추가 유틸리티 함수들 =====

void print_sensor_debug(const SensorData& data) {
    // 디버그용 센서 값 출력
    Serial.println("=== 센서 디버그 정보 ===");
    Serial.printf("Flex: %.1f, %.1f, %.1f, %.1f, %.1f\n", 
                  data.flex_1, data.flex_2, data.flex_3, data.flex_4, data.flex_5);
    Serial.printf("Gyro: %.2f, %.2f, %.2f\n", 
                  data.gyro_x, data.gyro_y, data.gyro_z);
    Serial.printf("Accel: %.2f, %.2f, %.2f\n", 
                  data.accel_x, data.accel_y, data.accel_z);
    Serial.printf("Battery: %.1f%%, Signal: %d dBm\n", 
                  data.battery_level, data.signal_strength);
    Serial.println("========================");
}

void calibrate_sensors() {
    // 센서 캘리브레이션 함수 (필요시 구현)
    Serial.println("센서 캘리브레이션 시작...");
    
    // 플렉스 센서 기준값 설정
    // IMU 센서 오프셋 조정
    
    Serial.println("센서 캘리브레이션 완료");
}

// ===== 에러 처리 =====

void handle_sensor_error(const String& error_msg) {
    Serial.println("센서 오류: " + error_msg);
    // 오류 복구 시도 또는 안전 모드 진입
}

void handle_communication_error(const String& error_msg) {
    Serial.println("통신 오류: " + error_msg);
    // 재연결 시도
} 