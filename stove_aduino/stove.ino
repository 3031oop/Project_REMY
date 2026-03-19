/* 

명령어 @FIRE@ON, @FIRE@OFF

*/

#include <SoftwareSerial.h>

SoftwareSerial BT(10, 11); // TX, RX
String inputString = "";

const int led2 = 2;
const int led3 = 3;
const int led4 = 4;

const int btn8 = 8;
const int btn9 = 9;

bool lastBtn8 = HIGH;
bool lastBtn9 = HIGH;

void setup() {
  Serial.begin(9600);
  BT.begin(9600);
  Serial.println("Bluetooth Ready");

  pinMode(led2, OUTPUT);
  pinMode(led3, OUTPUT);
  pinMode(led4, OUTPUT);

  pinMode(btn8, INPUT_PULLUP);
  pinMode(btn9, INPUT_PULLUP);

  // 초기 상태: 모두 끄기
  allLED(LOW);
}

void loop() {
  // 1. 블루투스 데이터 수신
  while (BT.available()) {
    char c = BT.read();
    Serial.write(c);
    inputString += c;

    if (c == '\n') {
      handleCommand(inputString);
      inputString = "";
    }
  }

  // 2. 물리 버튼 체크
  bool currentBtn8 = digitalRead(btn8);
  bool currentBtn9 = digitalRead(btn9);

  if (lastBtn8 == HIGH && currentBtn8 == LOW) { // 8번 버튼: ON
    Serial.println("불 켜기");
    allLED(HIGH);
    delay(50);
  }
  if (lastBtn9 == HIGH && currentBtn9 == LOW) { // 9번 버튼: OFF
    Serial.println("불 끄기");
    allLED(LOW);
    delay(50);
  }
  lastBtn8 = currentBtn8;
  lastBtn9 = currentBtn9;

  // 3. 디버깅용 (시리얼 -> 블루투스)
  while (Serial.available()) {
    char c = Serial.read();
    BT.write(c);
  }
}

// LED 전체 제어 함수
void allLED(int mode) {
  digitalWrite(led2, mode);
  digitalWrite(led3, mode);
  digitalWrite(led4, mode);
}

// 서버 명령어 처리
void handleCommand(String cmd) {
  cmd.trim();
  Serial.print("\n[명령 분석]: ");
  Serial.println(cmd);

  if (cmd.indexOf("@FIRE@ON") != -1) {
    Serial.println("  FIRE ON  ");
    allLED(HIGH);
  } 
  else if (cmd.indexOf("@FIRE@OFF") != -1) {
    Serial.println(" FIRE OFF  ");
    allLED(LOW);
  }
}
