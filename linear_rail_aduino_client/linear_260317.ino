/* 2026. 3. 16 
1. 와이파이 서버 ROBOTA 자동 연결 (5G는 와이파이 모듈 스펙상 불가)
2. TCP 10.10.141.43:5000에 자동 접속 ID:PW = LR:1234
3. 명령어 : 1) 좌표(X,Y,Z), 2)긴급 정지 @STOP, 3)원점 복귀 @ORIGIN
            ex) 100, 0, 0 좌표 입력시 Origin 위치에서 우측으로 100mm 이동
4. 양측 리미트 스위치 간 실제 간격 308mm , 브라켓 가로 폭 60mm
  가동범위 248 = (리미트 스위치 간 실제간격) 308 - (브라켓 가로 폭) 60

*추후 추가 해야 할 것*
- OMX와 결합 후 테스트하며 좌표값 디테일하게 맞추기
- TTS로 제어 (부가적인 부분) 
*/

#include <HCMotor.h>
#include <SoftwareSerial.h>
#include <math.h>

#define DIR_PIN 8 
#define CLK_PIN 9 
#define LLIMIT_PIN 3
#define RLIMIT_PIN 2

SoftwareSerial espSerial(10, 11); 
HCMotor HCMotor;

// 전역 변수 설정
int Speed = 4;
float currentX = 0;
float targetX = 0, targetY = 0, targetZ = 0;
float residualX = 0;
bool isMoving = false;

// ===== 좌표 변환 보정 파라미터 (환경에 맞게 수정) 260317 =====
const float THETA_DEG = 10.0;     // 카메라 기울기 각도
const float T_BA[3] = {0.0, 0.0, 0.0}; // 레일 원점에서 본 카메라 원점 오프셋 {tx, ty, tz}

// 레일 물리 설정
const float LIMIT_TO_LIMIT = 308.0;   // 리미트 스위치 간 실제 거리
const float BRACKET_WIDTH = 60.0;     // 브라켓(뭉치)의 가로 폭
const float MAX_RAIL_LENGTH = LIMIT_TO_LIMIT - BRACKET_WIDTH; // 250.0mm
const float L_OFFSET = 5.0; 
const float SAFE_MAX = MAX_RAIL_LENGTH - 5.0; // 여유 공간 확보
const float STEPS_PER_MM = 38.0; 

// 함수 선언 (프로토타입)
bool waitForResponse(String target, int timeout);
void connectWiFiAndServer();
void homing();
void updateMotorDrive();
void stopMotor();
void parseValues(String data);
void sendCompleteMessage();
// 함수 추가 (OMX 좌표계 변환)
void A_to_B(const float pA[3], float theta_deg, const float tBA[3], float pB[3]);

void setup() {
  Serial.begin(9600);
  espSerial.begin(9600);

  HCMotor.Init();
  HCMotor.attach(0, STEPPER, CLK_PIN, DIR_PIN);
  HCMotor.DutyCycle(0, 0);

  pinMode(LLIMIT_PIN, INPUT_PULLUP); 
  pinMode(RLIMIT_PIN, INPUT_PULLUP); 

  Serial.println("--- Starting System ---");
  homing();
  connectWiFiAndServer();
}

void loop() {
  if (espSerial.available()) {
    espSerial.setTimeout(50);
    String incomingData = espSerial.readStringUntil('\n');
    incomingData.trim();
    
    if (incomingData.length() > 0) {
      if (incomingData.indexOf("CLOSED") != -1) {
        connectWiFiAndServer();
        return;
      }

      int dataStartIndex = incomingData.lastIndexOf(']') + 1;
      String cleanData = (dataStartIndex > 0) ? incomingData.substring(dataStartIndex) : incomingData;
      cleanData.trim();

      if (cleanData.indexOf('@') != -1) {
        if (cleanData.indexOf("STOP") != -1) stopMotor();
        else if (cleanData.indexOf("ORIGIN") != -1) {
          homing();
          sendCompleteMessage();
        }
      } else if (cleanData.indexOf(',') != -1) {
        parseValues(cleanData);
        isMoving = true; 
      }
    }
  }
  updateMotorDrive();
}

void updateMotorDrive() {
  if (!isMoving) return;

  bool hitLeft = (digitalRead(LLIMIT_PIN) == LOW);
  bool hitRight = (digitalRead(RLIMIT_PIN) == LOW);

  if (hitLeft || hitRight) {
    HCMotor.DutyCycle(0, 0); 
    isMoving = false;
    delay(1000); // 정지 대기

    // [핵심] 리미트를 친 순간, 못다 한 거리를 계산 (예: 300 - 247 = 53)
    residualX = targetX - currentX; 
    if (residualX < 0) residualX = 0; // 혹시 마이너스가 나오면 0 처리

    Serial.print("Limit Hit! Current: "); Serial.print(currentX);
    Serial.print(" | Target was: "); Serial.print(targetX);
    Serial.print(" | Residual: "); Serial.println(residualX);

    // 탈출 로직 (이전과 동일)
    int escapeDir = hitLeft ? REVERSE : FORWARD;
    HCMotor.Direction(0, escapeDir);
    unsigned long escapeSteps = (unsigned long)(15.0 * STEPS_PER_MM);
    HCMotor.Steps(0, escapeSteps);
    HCMotor.DutyCycle(0, Speed);
    delay(2000); 
    HCMotor.DutyCycle(0, 0);

    // 좌표 동기화 (이제 더 이상 갈 수 없는 물리적 끝점임을 명시)
    if (hitLeft) currentX = 0; 
    else currentX = MAX_RAIL_LENGTH; // 가동범위 끝(250)으로 설정
    
    targetX = currentX; 
    sendCompleteMessage(); // 여기서 서버로 residualX(53)가 전송됩니다.
    return;
  }

  // [주행 로직]
  float distanceToMove = targetX - currentX;
  float absDiff = abs(distanceToMove);

  if (absDiff > 0.5) { 
    int dir = (targetX > currentX) ? REVERSE : FORWARD; 
    float currentFactor = (dir == REVERSE) ? 9.6 : 9.6; 

    HCMotor.Direction(0, dir);
    HCMotor.Steps(0, CONTINUOUS);
    HCMotor.DutyCycle(0, Speed);
    
    delay(20); 
    float stepPerLoop = currentFactor / STEPS_PER_MM;
    
    if (dir == REVERSE) currentX += stepPerLoop;
    else currentX -= stepPerLoop;
  } else {
    stopMotor();
    sendCompleteMessage();
  }
}

void homing() {
  Serial.println("--- Homing ---");
  HCMotor.Direction(0, FORWARD);
  HCMotor.Steps(0, CONTINUOUS);
  HCMotor.DutyCycle(0, Speed);
  while(digitalRead(LLIMIT_PIN) == HIGH);
  
  HCMotor.DutyCycle(0, 0); 
  delay(1000); 
  
  HCMotor.Direction(0, REVERSE);
  HCMotor.Steps(0, (unsigned long)(L_OFFSET * STEPS_PER_MM));
  HCMotor.DutyCycle(0, Speed);
  delay(2500); 
  
  HCMotor.DutyCycle(0, 0);
  currentX = 0; 
  targetX = 0;
  isMoving = false;
  Serial.println("Homing Done.");
}

// ===== 좌표 변환 함수 (카메라 좌표 pA -> 레일 좌표 pB) =====
void A_to_B(const float pA[3], float theta_deg, const float tBA[3], float pB[3]) {
  float theta = theta_deg * PI / 180.0;

  float xA = pA[0];
  float yA = pA[1];
  float zA = pA[2];

  float tx = tBA[0];
  float ty = tBA[1];
  float tz = tBA[2];

  // 계산 공식 적용
  pB[0] = -xA + tx;

  // yB = yA * sin(theta) - zA * cos(theta)
  pB[1] = (yA * sin(theta)) - (zA * cos(theta)) + ty;

  // zB = -yA * cos(theta) - zA * sin(theta)
  pB[2] = (-yA * cos(theta)) - (zA * sin(theta)) + tz;
}

void parseValues(String data) {
  float pA[3] = {0, 0, 0};
  float pB[3] = {0, 0, 0};

  int firstComma = data.indexOf(',');
  int secondComma = data.indexOf(',', firstComma + 1);
  
  if (firstComma > 0 && secondComma > firstComma) {
    pA[0] = data.substring(0, firstComma).toFloat();            // 입력받은 X
    pA[1] = data.substring(firstComma + 1, secondComma).toFloat(); // 입력받은 Y
    pA[2] = data.substring(secondComma + 1).toFloat();          // 입력받은 Z

    // 좌표 보정 실행
    A_to_B(pA, THETA_DEG, T_BA, pB);

    // 보정된 pB[0] 값을 레일의 목표 X축으로 설정
    targetX = constrain(pB[1], 0, MAX_RAIL_LENGTH);
    targetY = pB[0]; // 추후 Y, Z 제어 모터 추가 시 사용
    targetZ = pB[2];

    residualX = 0; 
    isMoving = true;

    Serial.print("Raw Y: "); Serial.print(pA[1]);
    Serial.print(" -> Transformed Target Y: "); Serial.println(targetY);
  }
}

bool waitForResponse(String target, int timeout) {
  unsigned long startTime = millis();
  String buffer = "";
  while (millis() - startTime < timeout) {
    while (espSerial.available()) {
      char c = espSerial.read();
      buffer += c;
      if (buffer.indexOf(target) != -1) return true;
    }
  }
  return false;
}

void connectWiFiAndServer() {
  Serial.println("Connecting WiFi...");
  espSerial.println("AT+CWMODE=1");
  waitForResponse("OK", 2000);
  espSerial.println("AT+CWJAP=\"robotA\",\"robotA1234\""); 
  if (waitForResponse("OK", 10000)) {
    Serial.println("Connecting Server...");
    espSerial.println("AT+CIPSTART=\"TCP\",\"10.10.141.43\",5000");
    if (waitForResponse("CONNECT", 5000)) {
      espSerial.println("AT+CIPSEND=10");
      if (waitForResponse(">", 2000)) espSerial.println("[LR:1234]");
    }
  }
}

void sendCompleteMessage() {
  String msg = "[OMXB]" + String(residualX, 0) + "," + String(targetY, 0) + "," + String(targetZ, 0) + "\n";
  espSerial.print("AT+CIPSEND=");
  espSerial.println(msg.length());
  if (waitForResponse(">", 1000)) espSerial.println(msg);
}

void stopMotor() {
  isMoving = false;
  targetX = currentX;
  HCMotor.DutyCycle(0, 0);
}