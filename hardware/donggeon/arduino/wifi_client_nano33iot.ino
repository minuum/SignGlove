/*
 * SignGlove WiFi 클라이언트 - Arduino Nano 33 IoT
 * 
 * 작성자: 양동건 (미팅 내용 반영)
 * 센서: Arduino LSM6DS3 (내장 IMU)
 * 통신: WiFi TCP 소켓
 * 
 * 주요 기능:
 * - Yaw 드리프트 필터링
 * - Roll/Pitch로 손바닥 방향 추정  
 * - 실시간 센서 데이터 전송
 */

#include <Arduino_LSM6DS3.h>
#include <WiFiNINA.h>

// ===== 설정 상수 =====
#define NUM_SAMPLES 10              // 평균 필터 샘플 수
#define GYRO_THRESHOLD 0.05         // 자이로 노이즈 임계값 (드리프트 방지)
#define LOOP_DELAY 100              // 메인 루프 지연 (100ms = 10Hz)
#define WIFI_SEND_INTERVAL 1000     // WiFi 전송 간격 (1초)

// ===== WiFi 설정 =====
char ssid[] = "SK_WiFiGIGA6828_2.4G";      // WiFi SSID
char pass[] = "JQF33@4624";                 // WiFi 비밀번호
char server[] = "192.168.45.23";            // 서버 IP 주소
int port = 5000;                             // 서버 포트

WiFiClient client;

// ===== 센서 변수 =====
float ax, ay, az;          // 가속도 값
float gx, gy, gz;          // 자이로 값
float yaw = 0.0;           // yaw 각도 (적분값)

// ===== 타이밍 변수 =====
unsigned long lastTime = 0;
unsigned long lastSendTime = 0;

void setup() {
    Serial.begin(9600);
    while (!Serial);
    
    Serial.println("=== SignGlove WiFi 클라이언트 시작 ===");
    
    // IMU 센서 초기화
    if (!IMU.begin()) {
        Serial.println("❌ IMU 초기화 실패!");
        while (1);
    }
    Serial.println("✅ IMU 센서 초기화 완료");
    
    // WiFi 연결
    connect_wifi();
    
    Serial.println("🚀 데이터 수집 시작...");
    lastTime = micros();
}

void loop() {
    // 센서 데이터 읽기 및 필터링
    read_and_filter_sensors();
    
    // Yaw, Pitch, Roll 계산
    calculate_orientation();
    
    // 시리얼 모니터 출력 (항상)
    print_sensor_data();
    
    // WiFi 전송 (주기적)
    send_wifi_data();
    
    delay(LOOP_DELAY);
}

void connect_wifi() {
    Serial.println("📶 WiFi 연결 중...");
    
    int status = WiFi.begin(ssid, pass);
    int attempts = 0;
    
    while (status != WL_CONNECTED && attempts < 20) {
        Serial.print(".");
        delay(2000);
        status = WiFi.begin(ssid, pass);
        attempts++;
    }
    
    if (status == WL_CONNECTED) {
        Serial.println("\n✅ WiFi 연결 성공!");
        Serial.print("IP 주소: ");
        Serial.println(WiFi.localIP());
        Serial.print("신호 강도: ");
        Serial.print(WiFi.RSSI());
        Serial.println(" dBm");
    } else {
        Serial.println("\n❌ WiFi 연결 실패!");
        while (1);
    }
}

void read_and_filter_sensors() {
    float axSum = 0, aySum = 0, azSum = 0;
    float gxSum = 0, gySum = 0, gzSum = 0;
    
    // 평균 필터 적용 (노이즈 감소)
    for (int i = 0; i < NUM_SAMPLES; i++) {
        while (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) {
            delay(1);
        }
        
        float temp_ax, temp_ay, temp_az;
        float temp_gx, temp_gy, temp_gz;
        
        IMU.readAcceleration(temp_ax, temp_ay, temp_az);
        IMU.readGyroscope(temp_gx, temp_gy, temp_gz);
        
        axSum += temp_ax;
        aySum += temp_ay; 
        azSum += temp_az;
        gxSum += temp_gx;
        gySum += temp_gy;
        gzSum += temp_gz;
        
        delay(2);
    }
    
    // 평균값 계산
    ax = axSum / NUM_SAMPLES;
    ay = aySum / NUM_SAMPLES;
    az = azSum / NUM_SAMPLES;
    gx = gxSum / NUM_SAMPLES;
    gy = gySum / NUM_SAMPLES;
    gz = gzSum / NUM_SAMPLES;
}

void calculate_orientation() {
    // 시간 차이 계산
    unsigned long now = micros();
    float dt = (now - lastTime) / 1e6;  // 초 단위 변환
    lastTime = now;
    
    // Yaw 적분 (드리프트 방지를 위한 임계값 적용)
    if (abs(gz) > GYRO_THRESHOLD) {
        yaw += gz * dt;
    }
    
    // Pitch/Roll 계산 (가속도 기반)
    float pitch = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0 / PI;
    float roll = atan2(ay, az) * 180.0 / PI;
    
    // 각도 범위 제한 (물리적 한계 적용)
    pitch = constrain(pitch, -90, 90);
    roll = constrain(roll, -180, 180);
    
    // Yaw 0~360도 범위로 정규화
    yaw = fmod(yaw + 360.0, 360.0);
    
    // 전역 변수에 저장 (다른 함수에서 사용)
    // pitch, roll은 지역 변수이므로 필요 시 전역 변수로 변경
}

void print_sensor_data() {
    // 실시간 모니터링을 위한 출력
    float pitch = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0 / PI;
    float roll = atan2(ay, az) * 180.0 / PI;
    
    Serial.print("Yaw: ");
    Serial.print(yaw, 2);
    Serial.print("° | Pitch: ");
    Serial.print(pitch, 2);
    Serial.print("° | Roll: ");
    Serial.print(roll, 2);
    Serial.print("° | WiFi: ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
}

void send_wifi_data() {
    // 전송 주기 체크
    if (millis() - lastSendTime < WIFI_SEND_INTERVAL) {
        return;
    }
    
    // Pitch/Roll 재계산 (전송용)
    float pitch = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0 / PI;
    float roll = atan2(ay, az) * 180.0 / PI;
    pitch = constrain(pitch, -90, 90);
    roll = constrain(roll, -180, 180);
    
    // 전송 데이터 구성 (CSV 형식)
    String data = String(millis()) + "," +
                  String(ax, 3) + "," + String(ay, 3) + "," + String(az, 3) + "," +
                  String(pitch, 2) + "," + String(roll, 2) + "," + String(yaw, 2) + "\n";
    
    // TCP 소켓으로 전송
    if (client.connect(server, port)) {
        client.print(data);
        client.stop();
        
        Serial.print("📤 전송: ");
        Serial.print(data);
        
        lastSendTime = millis();
    } else {
        Serial.println("❌ 서버 연결 실패");
    }
}

// ===== 추가 유틸리티 함수 =====

void reset_yaw() {
    // Yaw 각도 초기화 (사용자 정의 0도 설정)
    yaw = 0.0;
    Serial.println("🔄 Yaw 각도 초기화");
}

void print_system_info() {
    // 시스템 정보 출력
    Serial.println("=== 시스템 정보 ===");
    Serial.print("WiFi SSID: ");
    Serial.println(ssid);
    Serial.print("서버 주소: ");
    Serial.print(server);
    Serial.print(":");
    Serial.println(port);
    Serial.print("샘플링 주기: ");
    Serial.print(LOOP_DELAY);
    Serial.println("ms");
    Serial.print("전송 주기: ");
    Serial.print(WIFI_SEND_INTERVAL);
    Serial.println("ms");
    Serial.println("==================");
}

void handle_serial_commands() {
    // 시리얼 명령어 처리 (디버깅용)
    if (Serial.available()) {
        String command = Serial.readString();
        command.trim();
        
        if (command == "reset") {
            reset_yaw();
        } else if (command == "info") {
            print_system_info();
        } else if (command == "help") {
            Serial.println("명령어: reset, info, help");
        }
    }
} 