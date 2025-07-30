/*
 * SignGlove UART 클라이언트 - 플렉스 센서 + IMU
 * 
 * 작성자: 양동건 (미팅 내용 반영)
 * 센서: 5개 플렉스 센서 + Arduino LSM6DS3 IMU
 * 통신: UART (시리얼 통신)
 * 
 * 주요 기능:
 * - 50Hz 샘플링 (20ms 간격)
 * - 1분 간격 일괄 전송
 * - 플렉스 센서 + 자이로 데이터
 */

#include <Arduino_LSM6DS3.h>

// ===== 플렉스 센서 핀 설정 =====
const int FLEX_PINS[5] = {A0, A1, A2, A3, A4};  // 5개 플렉스 센서

// ===== 타이밍 설정 =====
const int SAMPLE_INTERVAL = 20;     // 50Hz = 20ms 간격
const int SAVE_INTERVAL = 60000;    // 1분 = 60000ms 간격

// ===== 버퍼 설정 =====
const int MAX_SAMPLES = 3000;       // 50Hz * 60s = 3000개 샘플
String buffer[MAX_SAMPLES];
int sampleCount = 0;

// ===== 타이밍 변수 =====
unsigned long lastSampleTime = 0;
unsigned long lastSaveTime = 0;

void setup() {
    Serial.begin(115200);
    while (!Serial);
    
    Serial.println("=== SignGlove UART 클라이언트 시작 ===");
    
    // IMU 센서 초기화
    if (!IMU.begin()) {
        Serial.println("❌ IMU 센서 초기화 실패!");
        while (1);
    }
    Serial.println("✅ IMU 센서 초기화 완료");
    
    // 플렉스 센서 핀 설정 (아날로그 입력은 별도 설정 불필요)
    Serial.println("✅ 플렉스 센서 준비 완료");
    
    // CSV 헤더 출력
    Serial.println("timestamp,pitch,roll,yaw,flex1,flex2,flex3,flex4,flex5");
    
    Serial.println("🚀 데이터 수집 시작 (50Hz)...");
}

void loop() {
    unsigned long now = millis();
    
    // 샘플링 주기 체크 (50Hz = 20ms)
    if (now - lastSampleTime >= SAMPLE_INTERVAL) {
        lastSampleTime = now;
        
        // 센서 데이터 읽기
        String sampleData = read_sensor_data(now);
        
        // 버퍼에 저장
        if (sampleCount < MAX_SAMPLES) {
            buffer[sampleCount++] = sampleData;
        }
        
        // 실시간 모니터링 (선택적)
        if (sampleCount % 50 == 0) {  // 1초마다 상태 출력
            Serial.print("📊 수집 중... ");
            Serial.print(sampleCount);
            Serial.print("/");
            Serial.print(MAX_SAMPLES);
            Serial.print(" (");
            Serial.print((now - lastSaveTime) / 1000);
            Serial.println("s)");
        }
    }
    
    // 1분마다 또는 버퍼가 가득 찰 때 일괄 전송
    if ((now - lastSaveTime >= SAVE_INTERVAL) || sampleCount >= MAX_SAMPLES) {
        send_buffered_data();
        
        // 버퍼 초기화
        sampleCount = 0;
        lastSaveTime = now;
        
        Serial.println("✅ 데이터 전송 완료. 새로운 수집 시작...");
    }
}

String read_sensor_data(unsigned long timestamp) {
    // IMU 데이터 읽기
    float gx, gy, gz;
    if (IMU.gyroscopeAvailable()) {
        IMU.readGyroscope(gx, gy, gz);
    } else {
        gx = gy = gz = 0.0;  // 센서 오류 시 기본값
    }
    
    // 자이로 데이터를 pitch, roll, yaw로 사용 (단순화)
    // 실제로는 더 복잡한 센서 융합이 필요하지만 미팅 코드 기준으로 단순화
    float pitch = gy;  // dps 단위
    float roll = gx;   // dps 단위  
    float yaw = gz;    // dps 단위
    
    // 플렉스 센서 읽기
    int flex[5];
    for (int i = 0; i < 5; i++) {
        flex[i] = analogRead(FLEX_PINS[i]);
    }
    
    // CSV 형식으로 데이터 구성
    String row = String(timestamp) + "," +
                 String(pitch, 2) + "," + String(roll, 2) + "," + String(yaw, 2);
    
    for (int i = 0; i < 5; i++) {
        row += "," + String(flex[i]);
    }
    
    return row;
}

void send_buffered_data() {
    Serial.println("📤 데이터 전송 시작...");
    Serial.println("timestamp,pitch,roll,yaw,flex1,flex2,flex3,flex4,flex5");
    
    // 버퍼의 모든 데이터를 시리얼로 전송
    for (int i = 0; i < sampleCount; i++) {
        Serial.println(buffer[i]);
        
        // 전송 중 지연 방지 (대용량 데이터)
        if (i % 100 == 0) {
            delay(1);  // 100개마다 1ms 대기
        }
    }
    
    Serial.println("📤 전송 완료!");
    Serial.print("총 ");
    Serial.print(sampleCount);
    Serial.println("개 샘플 전송됨");
}

// ===== 고급 센서 융합 함수 (미래 확장용) =====

void read_advanced_imu_data(float& pitch, float& roll, float& yaw) {
    // 가속도 + 자이로 융합을 통한 정확한 각도 계산
    // 현재는 단순화된 버전 사용
    
    float ax, ay, az;
    float gx, gy, gz;
    
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
        IMU.readAcceleration(ax, ay, az);
        IMU.readGyroscope(gx, gy, gz);
        
        // 가속도 기반 pitch/roll 계산
        pitch = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0 / PI;
        roll = atan2(ay, az) * 180.0 / PI;
        
        // 자이로 기반 yaw 추정 (적분 필요, 드리프트 보정 필요)
        yaw = gz;  // 간단한 버전
        
    } else {
        pitch = roll = yaw = 0.0;
    }
}

int read_filtered_flex(int pin, int samples = 5) {
    // 플렉스 센서 노이즈 필터링
    long sum = 0;
    for (int i = 0; i < samples; i++) {
        sum += analogRead(pin);
        delay(1);
    }
    return sum / samples;
}

void calibrate_flex_sensors() {
    // 플렉스 센서 캘리브레이션 (손가락 펴기/굽히기 기준점 설정)
    Serial.println("🔧 플렉스 센서 캘리브레이션 시작...");
    Serial.println("모든 손가락을 펴고 엔터를 누르세요.");
    
    while (!Serial.available()) {
        delay(100);
    }
    Serial.read(); // 엔터 키 소모
    
    // 펼친 상태 기준값 읽기
    int extended_values[5];
    for (int i = 0; i < 5; i++) {
        extended_values[i] = read_filtered_flex(FLEX_PINS[i]);
        Serial.print("플렉스 ");
        Serial.print(i + 1);
        Serial.print(" 펼친 상태: ");
        Serial.println(extended_values[i]);
    }
    
    Serial.println("모든 손가락을 굽히고 엔터를 누르세요.");
    while (!Serial.available()) {
        delay(100);
    }
    Serial.read(); // 엔터 키 소모
    
    // 굽힌 상태 기준값 읽기
    int flexed_values[5];
    for (int i = 0; i < 5; i++) {
        flexed_values[i] = read_filtered_flex(FLEX_PINS[i]);
        Serial.print("플렉스 ");
        Serial.print(i + 1);
        Serial.print(" 굽힌 상태: ");
        Serial.println(flexed_values[i]);
    }
    
    Serial.println("✅ 캘리브레이션 완료!");
}

void print_real_time_data() {
    // 실시간 센서 값 모니터링
    float gx, gy, gz;
    IMU.readGyroscope(gx, gy, gz);
    
    Serial.print("Gyro(dps): ");
    Serial.print(gx, 2);
    Serial.print(", ");
    Serial.print(gy, 2);
    Serial.print(", ");
    Serial.print(gz, 2);
    
    Serial.print(" | Flex: ");
    for (int i = 0; i < 5; i++) {
        Serial.print(analogRead(FLEX_PINS[i]));
        if (i < 4) Serial.print(", ");
    }
    Serial.println();
}

// ===== 에러 처리 및 복구 =====

void handle_sensor_error() {
    Serial.println("❌ 센서 읽기 오류 발생");
    
    // IMU 재초기화 시도
    if (!IMU.begin()) {
        Serial.println("⚠️ IMU 재초기화 실패");
    } else {
        Serial.println("✅ IMU 재초기화 성공");
    }
}

void emergency_data_dump() {
    // 긴급 상황 시 현재 버퍼 데이터 즉시 전송
    Serial.println("🚨 긴급 데이터 전송!");
    send_buffered_data();
    sampleCount = 0;
} 