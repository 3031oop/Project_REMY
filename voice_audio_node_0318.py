import time
import queue
import socket
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import keyboard
import os
from faster_whisper import WhisperModel

try:
    import winsound
except ImportError:
    winsound = None


# =========================
# 네트워크 설정
# =========================
SERVER_HOST = "10.10.141.50"   # PC-A TCP Server IP로 수정
SERVER_PORT = 5000

CLIENT_ID = "VOI"
PASSWORD = "PASSWD"

# 현재 팀 구조 호환용 ID
WA_TARGET_ID = "WA"        # 터틀봇 브리지 ID
MAIN_TARGET_ID = "OMXA"        # 상위 판단/브리지 쪽 ID (필요시 EYE/OMXA로 조정)


# =========================
# STT / 오디오 설정
# =========================
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"

MODEL_SIZE = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

MIN_RECORD_SEC = 0.4
SLEEP_SEC = 0.02

AUDIO_DIR = Path(r"C:\Users\KCCISTC\OneDrive\Desktop\굿")
#AUDIO_DIR = Path("/home/ubuntu/remy_wav")


# =========================
# 오디오 우선순위
# =========================
PRIORITY_MAP = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}


# =========================
# 상태 변수
# =========================
is_recording = False
audio_chunks = []

audio_queue = queue.Queue()
audio_lock = threading.Lock()
current_audio_priority = -1
is_audio_playing = False

sock = None
recv_stop = False


# =========================
# 이벤트 설명
# =========================
EVENT_DESC_MAP = {
    "EV_START": "시작 이벤트",
    "EV_CHECK_REQUEST": "확인 요청 이벤트",
    "EV_CONFIRM_DONE": "확인 완료 이벤트",
    "EV_END": "종료 이벤트",
    "EV_STOP": "정지 이벤트",
    "EV_SALT": "소금 요청 이벤트",
    "EV_PEPPER": "후추 요청 이벤트",
    "EV_SUGAR": "설탕 요청 이벤트",
    "EV_UNKNOWN": "알 수 없는 명령",
}


def event_description(event_code: str) -> str:
    return EVENT_DESC_MAP.get(event_code, "알 수 없는 이벤트")


# =========================
# TCP 수신 payload -> wav 매핑
# 서버/브리지에서 [SENDER]payload 형태로 들어오는 payload 기준
# =========================
PAYLOAD_AUDIO_MAP = {
    # 시스템
    "system_start": ("system_start.wav", "LOW"),
#"system_finish": ("system_finish.wav", "LOW"),
    "system_idle": ("system_idle.wav", "LOW"),
#"system_auto_return": ("system_auto_return.wav", "MEDIUM"),

    # 위험 / 조리
    "danger": ("danger.wav", "CRITICAL"),
    "detect_obj": ("detect.wav", "MEDIUM"),
    "cook_safe": ("cook_safe.wav", "LOW"),
    "cook_object": ("cook_object.wav", "LOW"),

    # 스토브
    "stove_start": ("stove_start.wav", "LOW"),
    "stove_user_left": ("stove_user_left.wav", "HIGH"),
    "stove_warning": ("stove_warning.wav", "CRITICAL"),

    # 재료
    "salt_move": ("salt_move.wav", "MEDIUM"),
    "pepper_move": ("pepper_move.wav", "MEDIUM"),
    "suger_move": ("suger_move.wav", "MEDIUM"),

    # 터틀봇
    "patrol": ("tb_patrol_start.wav", "MEDIUM"),
    "detect": ("tb_object_found.wav", "HIGH"),
    "depart": ("tb_location.wav", "HIGH"),
    "wait_return": ("tb_confirm.wav", "HIGH"),
    "force_return": ("tb_auto_return.wav", "HIGH"),
    "return": ("tb_return.wav", "MEDIUM"),

    # 에러
    "error_voice": ("error_voice.wav", "MEDIUM"),
    "error_comm": ("error_comm.wav", "HIGH"),
    "error_sensor": ("error_sensor.wav", "HIGH"),
    "error_safe_stop": ("error_safe_stop.wav", "CRITICAL"),
}


# =========================
# 경로 함수
# =========================
def get_audio_path(filename: str):
    file_path = AUDIO_DIR / filename
    if file_path.exists():
        return file_path
    return None


# =========================
# 오디오 콜백
# =========================
def audio_callback(indata, frames, time_info, status):
    global is_recording, audio_chunks

    if status:
        print(f"[오디오 상태] {status}")

    if is_recording:
        audio_chunks.append(indata.copy())


# =========================
# 텍스트 정규화
# =========================
def normalize_text(text: str) -> str:
    t = text.strip().lower()
    t = t.replace(" ", "")
    return t


# =========================
# STT 명령 매핑
# =========================
def map_command(text: str):
    t = normalize_text(text)

    stop_tokens = ["멈춰", "그만", "정지", "멈추어"]
    end_tokens = ["종료", "끝낼", "끝났", "끝", "다했", "다했어"]
    salt_tokens = ["소금", "소금줘", "소금가져", "소금가져다줘"]
    pepper_tokens = ["후추", "후추줘", "후추가져", "후추가져다줘"]
    sugar_tokens = ["설탕", "설탕줘", "설탕가져", "설탕가져다줘"]
    start_strong_tokens = ["시작"]
    start_weak_tokens = ["조리", "요리"]
    check_request_tokens = ["떨어", "도와", "찾아", "확인해", "뭐있", "뭐있어", "뭐있나"]
    confirm_done_tokens = ["오케이", "알았", "확인됐", "됐다", "복귀", "돌아가", "복기"]

    if any(token in t for token in stop_tokens):
        return "EV_STOP"

    if any(token in t for token in end_tokens):
        return "EV_END"

    if any(token in t for token in salt_tokens):
        return "EV_SALT"

    if any(token in t for token in pepper_tokens):
        return "EV_PEPPER"

    if any(token in t for token in sugar_tokens):
        return "EV_SUGAR"

    if any(token in t for token in start_strong_tokens):
        return "EV_START"

    if any(token in t for token in start_weak_tokens):
        return "EV_START"

    if any(token in t for token in check_request_tokens):
        return "EV_CHECK_REQUEST"

    if any(token in t for token in confirm_done_tokens):
        return "EV_CONFIRM_DONE"

    return "EV_UNKNOWN"


# =========================
# WAV 재생
# =========================
def play_wav_blocking(file_path: Path):
    if winsound is None:
        print("[오디오 에러] winsound 사용 불가")
        return

    winsound.PlaySound(str(file_path), winsound.SND_FILENAME)

#def play_wav_blocking(file_path: Path):
#    # 리눅스 전용: ffplay를 사용하여 소리 재생
#    # -nodisp: 창 안 띄움, -autoexit: 재생 후 종료
#    os.system(f"ffplay -nodisp -autoexit {file_path} > /dev/null 2>&1")

# =========================
# 오디오 큐 등록
# =========================
def enqueue_audio_from_payload(payload: str):
    payload = payload.strip()

    if payload not in PAYLOAD_AUDIO_MAP:
        print(f"[오디오] 매핑 없음: {payload}")
        return

    filename, priority_name = PAYLOAD_AUDIO_MAP[payload]
    priority = PRIORITY_MAP[priority_name]
    file_path = get_audio_path(filename)

    if file_path is None:
        print(f"[오디오] 파일 없음: {filename}")
        return

    audio_queue.put((priority, payload, file_path))


# =========================
# 오디오 워커
# =========================
def audio_worker():
    global current_audio_priority, is_audio_playing

    while True:
        priority, event_code, file_path = audio_queue.get()

        try:
            skip = False

            with audio_lock:
                if is_audio_playing and priority < current_audio_priority:
                    print(f"[오디오] 더 높은 우선순위 재생 중, 무시: {event_code}")
                    skip = True
                else:
                    is_audio_playing = True
                    current_audio_priority = priority

            if skip:
                audio_queue.task_done()
                continue

            print(f"[오디오 재생] {event_code} -> {file_path.name}")
            play_wav_blocking(file_path)

        except Exception as e:
            print(f"[오디오 에러] {e}")

        finally:
            with audio_lock:
                is_audio_playing = False
                current_audio_priority = -1

            audio_queue.task_done()


# =========================
# 서버 송신
# 현재 서버 형식 호환: [TO]payload
# =========================
def send_wire_message(target_id: str, payload: str):
    global sock

    if sock is None:
        print("[송신 실패] 소켓 없음")
        return

    try:
        msg = f"[{target_id}]{payload}\n"
        sock.sendall(msg.encode("utf-8"))
        print(f"[SEND] {msg.strip()}")
    except Exception as e:
        print(f"[송신 에러] {e}")
        enqueue_audio_from_payload("error_comm")


# =========================
# 이벤트 처리
# 현재 팀 구조에 맞게:
# - EV_CHECK_REQUEST -> WA patrol 시작 (tts1)
# - EV_CONFIRM_DONE -> WA return 시작 (tts2)
# - 나머지 시스템 이벤트는 MAIN_TARGET_ID로 전달
# =========================
def dispatch_event(event_code: str):
    if event_code == "EV_UNKNOWN":
        enqueue_audio_from_payload("error_voice")
        return

    if event_code == "EV_CHECK_REQUEST":
        # 터틀봇 순찰 시작
        send_wire_message(WA_TARGET_ID, "tts1")
        return

    if event_code == "EV_CONFIRM_DONE":
        # 터틀봇 복귀 시작
        send_wire_message(WA_TARGET_ID, "tts2")
        return

    # 나머지는 상위 판단 쪽으로 전달
    send_wire_message(MAIN_TARGET_ID, event_code.lower())


# =========================
# 서버 수신
# 서버가 주는 메시지 형식: [SENDER]payload
# ex) [HSJ_WF]detect
# =========================
def recv_loop():
    global recv_stop, sock

    buffer = ""

    while not recv_stop:
        try:
            data = sock.recv(4096)
            if not data:
                print("[소켓] 서버 연결 종료")
                break

            buffer += data.decode("utf-8", errors="ignore")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                print(f"[RECV] {line}")
                handle_server_message(line)

        except Exception as e:
            print(f"[수신 에러] {e}")
            enqueue_audio_from_payload("error_comm")
            break


def handle_server_message(line: str):
    # 기대 형식: [SENDER]payload
    try:
        if "]" not in line or "[" not in line:
            return

        open_idx = line.find("[")
        close_idx = line.find("]")

        sender = line[open_idx + 1:close_idx].strip()
        payload = line[close_idx + 1:].strip()

        if not payload:
            return

        # 현재는 payload 기반으로 바로 오디오 재생
        # ex) detect, depart, wait_return, force_return, patrol, return ...
        enqueue_audio_from_payload(payload)

    except Exception as e:
        print(f"[메시지 파싱 에러] {e}")


# =========================
# Whisper 추론
# =========================
def transcribe_audio(model: WhisperModel, audio_data: np.ndarray):
    if audio_data is None or len(audio_data) == 0:
        print("[결과] 녹음된 데이터가 없음")
        enqueue_audio_from_payload("error_voice")
        return

    print("[상태] 음성 인식 중...")

    segments, info = model.transcribe(
        audio_data,
        language="ko",
        vad_filter=True
    )

    text = "".join(seg.text for seg in segments).strip()

    if not text:
        print("[원문] 인식 결과 없음")
        enqueue_audio_from_payload("error_voice")
        return

    print(f"[원문] {text}")

    event_code = map_command(text)
    print(f"[이벤트] {event_code} ({event_description(event_code)})")

    dispatch_event(event_code)


# =========================
# 폴더/파일 체크
# =========================
def validate_audio_dir():
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"[오디오 폴더 없음] {AUDIO_DIR}")

    print(f"[오디오 폴더 확인] {AUDIO_DIR}")

    missing = []
    for _, (filename, _) in PAYLOAD_AUDIO_MAP.items():
        if not (AUDIO_DIR / filename).exists():
            missing.append(filename)

    if missing:
        print("[주의] 아래 오디오 파일이 없습니다:")
        for m in sorted(set(missing)):
            print(f" - {m}")
    else:
        print("[오디오 파일 확인] 필수 파일 존재")


# =========================
# 서버 연결
# =========================
def connect_server():
    global sock

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_HOST, SERVER_PORT))

    auth_msg = f"[{CLIENT_ID}:{PASSWORD}]"
    sock.sendall(auth_msg.encode("utf-8"))
    print(f"[서버 인증] {auth_msg}")


# =========================
# 메인
# =========================
def main():
    global is_recording, audio_chunks, recv_stop

    validate_audio_dir()

    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    audio_thread.start()

    connect_server()
    threading.Thread(target=recv_loop, daemon=True).start()

    print("[1] Whisper 모델 로딩 시작...")
    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )
    print("[2] Whisper 모델 로딩 완료")

    print("\n사용법:")
    print("- 스페이스바 누르고 있는 동안만 녹음")
    print("- 스페이스바 떼면 인식 후 서버로 명령 송신")
    print("- 서버/장치 상태 메시지를 받으면 WAV 재생")
    print("- ESC 누르면 종료\n")

    prev_space_state = False
    record_start_time = None

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=audio_callback
    ):
        print("[3] 마이크 입력 대기 중...\n")

        while True:
            try:
                if keyboard.is_pressed("esc"):
                    print("\n[종료] 프로그램을 종료합니다.")
                    recv_stop = True
                    break

                current_space_state = keyboard.is_pressed("space")

                if current_space_state and not prev_space_state:
                    audio_chunks = []
                    is_recording = True
                    record_start_time = time.time()
                    print("[녹음] 시작...")

                elif not current_space_state and prev_space_state:
                    is_recording = False
                    duration = time.time() - record_start_time if record_start_time else 0.0
                    print(f"[녹음] 종료 (길이: {duration:.2f}초)")

                    if duration < MIN_RECORD_SEC:
                        print("[경고] 너무 짧게 눌렀음")
                        enqueue_audio_from_payload("error_voice")
                    elif len(audio_chunks) == 0:
                        print("[경고] 녹음된 음성이 없음")
                        enqueue_audio_from_payload("error_voice")
                    else:
                        audio_data = np.concatenate(audio_chunks, axis=0).reshape(-1)
                        transcribe_audio(model, audio_data)
                        print()

                    record_start_time = None

                prev_space_state = current_space_state
                time.sleep(SLEEP_SEC)

            except KeyboardInterrupt:
                print("\n[종료] Ctrl+C 입력으로 종료합니다.")
                recv_stop = True
                break
            except Exception as e:
                print(f"[에러] {e}")
                time.sleep(0.2)

    try:
        if sock:
            sock.close()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[최상위 에러] {e}")
        input("엔터 누르면 종료")
 
