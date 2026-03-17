#client program 
#kcci iot/embedded 
import socket 
import threading 
import time
import RPi.GPIO as GPIO
import re
import sys

# 하드웨어 설정
LED_PINS = [2,3,4,14,15,18,17,27,22]
VIB_PINS = [24,10,9,25,11,8,7,5] # 진동 추가 260317 성균 수정
BUTTON_PIN = 21

# 소켓 설정 (전역변수로 옮김 260317 성균수정)
HOST = "10.10.141.51" 
PORT = 5000 
ADDR = (HOST,PORT)
recvFlag = False
rsplit = []
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 인식 대상 리스트
TOOLS_NAME = ["Knife", "Fork", "Ladle", "Plate"]
TARGET = 0

# ===== 버튼 콜백 함수 =====
def button_callback():
	global TARGET, TOOLS_NAME, s
	prev_state = GPIO.LOW
	print(">>> 버튼 모니터링")

	while True:
		current_state = GPIO.input(BUTTON_PIN)

		if current_state == GPIO.HIGH and prev_state == GPIO.LOW:
			TARGET = (TARGET + 1) % len(TOOLS_NAME)
			print(f"\n[버튼 클릭] 타켓 변경: {TOOLS_NAME[TARGET]} (Index: {TARGET})")

			try: 
				msg = f"[OMXA]TARGET@{TARGET}\n"
				s.send(msg.encode())
			except:
				print("서버로 타켓 변경 메시지 전송 실패")
			# 디바운싱
			time.sleep(0.2)

		prev_state = current_state
		time.sleep(0.01)

# gpio 모드 설정
GPIO.setmode(GPIO.BCM)
# 경고 메시지 끄기 
GPIO.setwarnings(False)

# 진동 추가 (260317 성균 수정)
for i in range(len(LED_PINS)):
	GPIO.setup(LED_PINS[i], GPIO.OUT)
	GPIO.setup(VIB_PINS[i], GPIO.OUT)
	GPIO.output(LED_PINS[i], False)
	GPIO.output(VIB_PINS[i], False)

# 버튼 설정 (내부 풀다운 저항 사용)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)


try:
	s.connect((HOST, PORT)) 
	def sendingMsg(): 
		s.send('[VI:PASSWD]'.encode()) 
		# time.sleep(0.5)
		while True: 
			time.sleep(5)
			# data = input() 
			# # data = bytes(data, "utf-8") 
			# data = bytes(data+'\n', "utf-8") 
			# s.send(data) 
		# s.close() 
	def gettingMsg(): 
		global rsplit
		global recvFlag
		while True: 
			data = s.recv(1024) 
			if not data:
				break
			
			rstr = data.decode("utf-8").upper().strip()
			# print(f"DEBUG: 수신된 원본 문자열 -> '{rstr}'")
			rsplit = re.split(r'[\]|\[@]|\n',rstr)  #'[',']','@' 분리
			
			recvFlag = True
			
			for pin in LED_PINS:
				GPIO.output(pin,False)
				
			if "LED@" in rstr:
				try:
					match = re.search(r'LED@(\d+)',rstr)
					if match:
						pin_num = int(match.group(1))
						if pin_num in LED_PINS:
							GPIO.output(pin_num, True)
							print(f">>> {pin_num}번 LED 점등")
				except: pass

			elif "DANGER" in rstr:
				GPIO.output(15, True)
				print(">>> [위험] 칼날 접근! 15번 LED 점등")
			elif "DETECTED" in rstr:
				for pin in LED_PINS:
					 GPIO.output(pin,True)
				print(">>> [감지] 모든 LED 점등")
				pass
			elif "OFF" in rstr:
				for pin in LED_PINS:
					GPIO.output(pin, False)
				print(">>> [안전] 상황 해제. LED 소등")
			#print('recv :',rsplit) 
		# s.close()


	threading.Thread(target=sendingMsg, daemon=True).start()
	threading.Thread(target=gettingMsg, daemon=True).start()
	threading.Thread(target=button_callback, daemon=True).start()
	print('connect is success')

	while True: 
		if recvFlag:
			# print('recv :',rsplit) 
			recvFlag = False
		time.sleep(0.1)
  
except Exception as e:
    print(f"Error: {e}")
	print('%s:%s'%ADDR)
	
finally:
	GPIO.cleanup()
	s.close()
	sys.exit()





