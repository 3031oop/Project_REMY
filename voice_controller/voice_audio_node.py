import time
import queue
import socket
import threading
from pathlib import Path
from collections import deque
import os

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

try:
    import winsound
except ImportError:
    winsound = None


# =========================
# 네트워크 설정
# =========================
SERVER_HOST = "10.10.141.126"
SERVER_PORT = 5000

CLIENT_ID = "VOI"
PASSWORD = "PASSWD"

WA_TARGET_ID = "WA"
MAIN_TARGET_ID = "EYE"
ALL_TARGET_ID = "ALLMSG"


# =========================
# STT / 오디오 설정
# =========================
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"

MODEL_SIZE = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

SLEEP_SEC = 0.02

AUDIO_DIR = Path(r"C:\Users\KCCISTC\Desktop\tts")


# =========================
# 호출어 / 연속듣기 설정
# =========================
WAKE_WORD_TOKENS = [
    "레미야",
    "래미야",
    "레미",
    "래미",
    "remy",
    "remiya",
    "애미야",
    "래이미아",
    "레이미야",
    "레몬",
    "네미야",
    "Let me",
    "Let me out",
    "리미야",
    "데미아",
    "레미아",
    "리미아",
    "르미아",
    "러미아",
    "데미야",
    "뇌미아",
    "네, 미아.",
    "네, 미아",
    "네, 미아?",
    "레미안",
    "렘이야",
    "러미아",
    "데뷔",
    "데뷔야",
    "LEMIAN",
    "뷰미아",
]

WAKE_CHECK_INTERVAL_SEC = 1.0
WAKE_WINDOW_SEC = 1.2
WAKE_COOLDOWN_SEC = 2.0

# 명령 녹음 관련
COMMAND_MAX_SEC = 6.5
COMMAND_SILENCE_SEC = 1.2
COMMAND_START_TIMEOUT_SEC = 5.0
MIN_COMMAND_SEC = 0.20

# 호출 직후 첫 단어가 잘리는 것 방지
COMMAND_PREROLL_SEC = 0.15

# 배경 잡음 환경 기준
RMS_SILENCE_THRESHOLD = 0.020

# RMS 디버그 출력
PRINT_RMS_DEBUG = True
RMS_PRINT_INTERVAL_SEC = 0.2

# Whisper 잡문장 필터
NOISE_TEXT_FILTERS = [
    "감사합니다",
    "수고하셨습니다",
    "네",
    "예",
    "알겠습니다",
    "고맙습니다",
    "네감사합니다",
    "네알겠습니다",
]


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
audio_queue = queue.Queue()
audio_lock = threading.Lock()
current_audio_priority = -1
is_audio_playing = False

sock = None
recv_stop = False

mic_lock = threading.Lock()
recent_audio_chunks = deque()
recent_audio_total_samples = 0

state_lock = threading.Lock()
listen_state = "WAIT_WAKE"   # WAIT_WAKE / WAKE_FEEDBACK / COMMAND_LISTENING
command_chunks = []
command_started_at = None
command_last_voice_at = None
command_has_voice = False
wake_last_detected_at = 0.0
last_wake_check_at = 0.0
last_rms_print_at = 0.0


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
    "EV_KNIFE": "칼 요청 이벤트",
    "EV_LADLE": "국자 요청 이벤트",
    "EV_FORK": "포크 요청 이벤트",
    "EV_PLATE": "접시 요청 이벤트",
    "EV_UNKNOWN": "알 수 없는 명령",
}


def event_description(event_code: str) -> str:
    return EVENT_DESC_MAP.get(event_code, "알 수 없는 이벤트")


# =========================
# TCP 수신 payload -> wav 매핑
# =========================
PAYLOAD_AUDIO_MAP = {
    # 시스템
    "system_start": ("system_start.wav", "LOW"),
    "system_idle": ("system_idle.wav", "LOW"),

    # 호출 피드백
    "wake_ack": ("remy_reback.wav", "MEDIUM"),

    # 위험 / 조리
    "knife_danger": ("danger.wav", "CRITICAL"),
    "knife_detected": ("detect.wav", "MEDIUM"),
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
    "sugar_move": ("suger_move.wav", "MEDIUM"),

    # 객체
    "knife_move": ("knife.wav", "MEDIUM"),
    "ladle_move": ("ladle.wav", "MEDIUM"),
    "fork_move": ("fork.wav", "MEDIUM"),
    "plate_move": ("plate.wav", "MEDIUM"),

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
# 텍스트 정규화
# =========================
def normalize_text(text: str) -> str:
    t = text.strip().lower()
    t = t.replace(" ", "")
    return t


def is_wake_word(text: str) -> bool:
    t = normalize_text(text)
    tokens = [normalize_text(x) for x in WAKE_WORD_TOKENS]
    return any(token in t for token in tokens)


def is_noise_text(text: str) -> bool:
    t = normalize_text(text)
    return t in [normalize_text(x) for x in NOISE_TEXT_FILTERS]


def strip_wake_word_prefix(text: str) -> str:
    t = text.strip()
    nt = normalize_text(t)

    for token in WAKE_WORD_TOKENS:
        n_token = normalize_text(token)
        if nt.startswith(n_token):
            stripped = t[len(token):].strip(" ,.!?")
            return stripped if stripped else t

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

    knife_tokens = ["칼", "칼줘", "칼가져", "칼가져다줘", "나이프", "knife"]
    ladle_tokens = ["국자", "국자줘", "국자가져", "국자가져다줘", "ladle"]
    fork_tokens = ["포크", "뽀크", "보컬", "포크줘", "포크가져", "포크가져다줘", "fork"]
    plate_tokens = ["접시", "탑시", "접시줘", "접시가져", "접시가져다줘", "플레이트", "plate"]

    start_strong_tokens = ["시작", "사장"]
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
    if any(token in t for token in knife_tokens):
        return "EV_KNIFE"
    if any(token in t for token in ladle_tokens):
        return "EV_LADLE"
    if any(token in t for token in fork_tokens):
        return "EV_FORK"
    if any(token in t for token in plate_tokens):
        return "EV_PLATE"
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

    try:
        winsound.PlaySound(str(file_path), winsound.SND_FILENAME)
    except Exception as e:
        print(f"[오디오 재생 에러] {e}")


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

    print(f"[오디오 큐] {payload} -> {filename} ({priority_name})")
    audio_queue.put((priority, payload, file_path))


def play_feedback_blocking(payload: str):
    if payload not in PAYLOAD_AUDIO_MAP:
        return

    filename, priority_name = PAYLOAD_AUDIO_MAP[payload]
    file_path = get_audio_path(filename)
    if file_path is None:
        print(f"[오디오] 파일 없음: {filename}")
        return

    print(f"[오디오 재생] {payload} -> {file_path.name}")
    with audio_lock:
        global is_audio_playing, current_audio_priority
        is_audio_playing = True
        current_audio_priority = PRIORITY_MAP[priority_name]

    try:
        play_wav_blocking(file_path)
    finally:
        with audio_lock:
            is_audio_playing = False
            current_audio_priority = -1


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
# =========================
def send_wire_message(target_id: str, payload: str):
    global sock

    if sock is None:
        print("[송신 실패] 소켓 없음")
        enqueue_audio_from_payload("error_comm")
        return False

    try:
        msg = f"[{target_id}]{payload}\n"
        sock.sendall(msg.encode("utf-8"))
        print(f"[SEND] {msg.strip()}")
        return True
    except Exception as e:
        print(f"[송신 에러] {e}")
        enqueue_audio_from_payload("error_comm")
        return False


# =========================
# 이벤트 처리
# =========================
def dispatch_event(event_code: str):
    if event_code == "EV_UNKNOWN":
        enqueue_audio_from_payload("error_voice")
        return

    if event_code == "EV_START":
        enqueue_audio_from_payload("system_start")
        send_wire_message(MAIN_TARGET_ID, "START")
        return

    if event_code == "EV_END":
        enqueue_audio_from_payload("system_idle")
        send_wire_message(MAIN_TARGET_ID, "FINISH")
        return

    if event_code == "EV_STOP":
        enqueue_audio_from_payload("error_safe_stop")
        send_wire_message(ALL_TARGET_ID, "STOP")
        return

    if event_code == "EV_SALT":
        enqueue_audio_from_payload("salt_move")
        send_wire_message(MAIN_TARGET_ID, "ID@1")
        return

    if event_code == "EV_PEPPER":
        enqueue_audio_from_payload("pepper_move")
        send_wire_message(MAIN_TARGET_ID, "ID@2")
        return

    if event_code == "EV_SUGAR":
        enqueue_audio_from_payload("suger_move")
        send_wire_message(MAIN_TARGET_ID, "ID@3")
        return

    if event_code == "EV_KNIFE":
        enqueue_audio_from_payload("knife_move")
        send_wire_message(MAIN_TARGET_ID, "OBJ@KNIFE")
        return

    if event_code == "EV_LADLE":
        enqueue_audio_from_payload("ladle_move")
        send_wire_message(MAIN_TARGET_ID, "OBJ@LADLE")
        return

    if event_code == "EV_FORK":
        enqueue_audio_from_payload("fork_move")
        send_wire_message(MAIN_TARGET_ID, "OBJ@FORK")
        return

    if event_code == "EV_PLATE":
        enqueue_audio_from_payload("plate_move")
        send_wire_message(MAIN_TARGET_ID, "OBJ@PLATE")
        return

    if event_code == "EV_CHECK_REQUEST":
        send_wire_message(WA_TARGET_ID, "tts1")
        return

    if event_code == "EV_CONFIRM_DONE":

        try:
            enqueue_audio_from_payload("return")
        except Exception as e:
            print(f"[피드백 재생 에러] {e}")

        enter_command_listening()
        send_wire_message(WA_TARGET_ID, "tts2")
        return


# =========================
# 서버 수신
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
    try:
        if "]" not in line or "[" not in line:
            return

        open_idx = line.find("[")
        close_idx = line.find("]")

        sender = line[open_idx + 1:close_idx].strip()
        payload = line[close_idx + 1:].strip()

        if not payload:
            return

        enqueue_audio_from_payload(payload)

    except Exception as e:
        print(f"[메시지 파싱 에러] {e}")


# =========================
# Whisper 공통 추론
# =========================
def transcribe_to_text(model: WhisperModel, audio_data: np.ndarray):
    if audio_data is None or len(audio_data) == 0:
        return ""

    try:
        segments, info = model.transcribe(
            audio_data,
            language="ko",
            vad_filter=True
        )
        text = "".join(seg.text for seg in segments).strip()
        return text
    except Exception as e:
        print(f"[Whisper 에러] {e}")
        return ""


def transcribe_and_dispatch_command(model: WhisperModel, audio_data: np.ndarray):
    if audio_data is None or len(audio_data) == 0:
        print("[결과] 녹음된 데이터가 없음")
        enqueue_audio_from_payload("error_voice")
        return

    print("[상태] 명령 음성 인식 중...")

    text = transcribe_to_text(model, audio_data)

    if not text:
        print("[원문] 인식 결과 없음")
        enqueue_audio_from_payload("error_voice")
        return

    text = strip_wake_word_prefix(text)
    print(f"[원문] {text}")

    if is_noise_text(text):
        print("[필터] 잡문장으로 판단되어 무시")
        enqueue_audio_from_payload("error_voice")
        return

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
# 최근 오디오 버퍼 관리
# =========================
def append_recent_audio(chunk_1d: np.ndarray):
    global recent_audio_total_samples

    now = time.time()

    with mic_lock:
        recent_audio_chunks.append((now, chunk_1d))
        recent_audio_total_samples += len(chunk_1d)

        keep_sec = max(WAKE_WINDOW_SEC + COMMAND_PREROLL_SEC + 1.0, 3.0)
        while recent_audio_chunks and (now - recent_audio_chunks[0][0] > keep_sec):
            _, old_chunk = recent_audio_chunks.popleft()
            recent_audio_total_samples -= len(old_chunk)
            if recent_audio_total_samples < 0:
                recent_audio_total_samples = 0


def get_recent_audio(seconds: float):
    now = time.time()
    out = []

    with mic_lock:
        for ts, chunk in recent_audio_chunks:
            if now - ts <= seconds:
                out.append(chunk)

    if not out:
        return np.array([], dtype=np.float32)

    return np.concatenate(out, axis=0).astype(np.float32)


# =========================
# 오디오 레벨
# =========================
def rms_level(x: np.ndarray) -> float:
    if x is None or len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


# =========================
# 상태 전환
# =========================
def enter_command_listening():
    global listen_state, command_chunks, command_started_at, command_last_voice_at, command_has_voice

    preroll_audio = get_recent_audio(COMMAND_PREROLL_SEC)

    with state_lock:
        listen_state = "COMMAND_LISTENING"
        command_chunks = []
        if len(preroll_audio) > 0:
            command_chunks.append(preroll_audio.astype(np.float32))
        command_started_at = time.time()
        command_last_voice_at = None
        command_has_voice = False

    print("[상태] 명령 입력 모드 시작")


def enter_wake_feedback():
    with state_lock:
        global listen_state
        listen_state = "WAKE_FEEDBACK"

    print("[상태] 호출어 감지 -> 피드백 재생")


def reset_to_wait_wake():
    global listen_state, command_chunks, command_started_at, command_last_voice_at, command_has_voice

    with state_lock:
        listen_state = "WAIT_WAKE"
        command_chunks = []
        command_started_at = None
        command_last_voice_at = None
        command_has_voice = False

    print("[상태] 호출 대기 모드")


def feedback_then_enter_command_listening():
    try:
        play_feedback_blocking("wake_ack")
    except Exception as e:
        print(f"[피드백 재생 에러] {e}")

    enter_command_listening()


# =========================
# 오디오 콜백
# =========================
def audio_callback(indata, frames, time_info, status):
    global command_chunks, command_last_voice_at, command_has_voice, last_rms_print_at

    if status:
        print(f"[오디오 상태] {status}")

    mono = indata.copy().reshape(-1).astype(np.float32)
    append_recent_audio(mono)

    current_rms = rms_level(mono)
    now = time.time()

    with state_lock:
        current_state = listen_state

        if current_state == "COMMAND_LISTENING":
            command_chunks.append(mono)

            if PRINT_RMS_DEBUG and (now - last_rms_print_at >= RMS_PRINT_INTERVAL_SEC):
                print(f"[RMS] {current_rms:.5f}")
                last_rms_print_at = now

            if current_rms >= RMS_SILENCE_THRESHOLD:
                command_last_voice_at = now
                command_has_voice = True


# =========================
# 호출어 검사
# =========================
def check_wake_word(model: WhisperModel):
    global wake_last_detected_at

    now = time.time()

    with audio_lock:
        if is_audio_playing:
            return

    with state_lock:
        if listen_state != "WAIT_WAKE":
            return

    if now - wake_last_detected_at < WAKE_COOLDOWN_SEC:
        return

    audio_data = get_recent_audio(WAKE_WINDOW_SEC)
    if len(audio_data) == 0:
        return

    text = transcribe_to_text(model, audio_data)

    if not text:
        return

    print(f"[호출어 검사] {text}")

    if is_wake_word(text):
        wake_last_detected_at = now
        enter_wake_feedback()
        threading.Thread(target=feedback_then_enter_command_listening, daemon=True).start()


# =========================
# 명령 종료 조건 검사
# =========================
def process_command_state(model: WhisperModel):
    with state_lock:
        current_state = listen_state
        started_at = command_started_at
        last_voice_at = command_last_voice_at
        has_voice = command_has_voice
        local_chunks = command_chunks[:]

    if current_state != "COMMAND_LISTENING" or started_at is None:
        return

    now = time.time()
    elapsed = now - started_at

    if not has_voice and elapsed > COMMAND_START_TIMEOUT_SEC:
        print("[명령] 호출 후 음성 없음 -> 종료")
        reset_to_wait_wake()
        return

    should_finish = False

    if elapsed >= COMMAND_MAX_SEC:
        print("[명령] 최대 녹음 길이 도달 -> 종료")
        should_finish = True

    if has_voice and last_voice_at is not None:
        if now - last_voice_at >= COMMAND_SILENCE_SEC:
            print("[명령] 무음 감지 -> 종료")
            should_finish = True

    if not should_finish:
        return

    reset_to_wait_wake()

    if not local_chunks:
        print("[명령] 수집된 오디오 없음")
        enqueue_audio_from_payload("error_voice")
        return

    audio_data = np.concatenate(local_chunks, axis=0).reshape(-1)
    duration = len(audio_data) / SAMPLE_RATE
    print(f"[명령] 최종 길이: {duration:.2f}초")

    if duration < MIN_COMMAND_SEC:
        print("[명령] 너무 짧은 명령")
        enqueue_audio_from_payload("error_voice")
        return

    transcribe_and_dispatch_command(model, audio_data)


# =========================
# 메인
# =========================
def main():
    global recv_stop, last_wake_check_at

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
    print("- 항상 마이크 입력을 듣습니다")
    print("- 1초마다 호출어(예: 레미야)를 검사합니다")
    print("- 호출어 감지 시 '네, 말씀하세요' 피드백을 재생합니다")
    print("- 피드백 재생이 끝난 뒤 명령 입력 모드로 전환됩니다")
    print("- 호출 후 최대 5초까지 사용자의 명령 시작을 기다립니다")
    print("- 말이 끝나고 1.2초 정도 조용하면 자동 인식합니다")
    print("- Ctrl+C 로 종료\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=audio_callback
    ):
        print("[3] 마이크 상시 대기 중...\n")

        while True:
            try:
                now = time.time()

                with state_lock:
                    current_state = listen_state

                if current_state == "WAIT_WAKE":
                    if now - last_wake_check_at >= WAKE_CHECK_INTERVAL_SEC:
                        last_wake_check_at = now
                        check_wake_word(model)

                elif current_state == "COMMAND_LISTENING":
                    process_command_state(model)

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
