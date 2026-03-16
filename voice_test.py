import time
import queue
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import keyboard
from faster_whisper import WhisperModel

try:
    import winsound
except ImportError:
    winsound = None


# =========================
# 설정값
# =========================
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"

MODEL_SIZE = "base"      # tiny / base / small
DEVICE = "cpu"           # cpu / cuda
COMPUTE_TYPE = "int8"    # cpu에서는 int8 권장

MIN_RECORD_SEC = 0.4
SLEEP_SEC = 0.02

# 반드시 네 환경에 맞게 수정
AUDIO_DIR = Path(r"C:\Users\KCCISTC\Desktop\finalproject_sound_voice\final_TTS")


# =========================
# 오디오 우선순위
# =========================
PRIORITY_MAP = {
    "CRITICAL": 3,
    "HIGH": 2,
    "MEDIUM": 1,
    "LOW": 0,
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
    "EV_PEPPER                                 ": "후추 요청 이벤트",
    "EV_SUGAR": "설탕 요청 이벤트",
    "EV_UNKNOWN": "알 수 없는 명령",
}


def event_description(event_code: str) -> str:
    return EVENT_DESC_MAP.get(event_code, "알 수 없는 이벤트")


# =========================
# 이벤트 -> 오디오 파일 매핑
# =========================
EVENT_AUDIO_MAP = {
    # 시스템
    "EV_START": ("system_start.wav", "LOW"),
    "EV_END": ("system_finish.wav", "LOW"),
    "EV_STOP": ("error_safe_stop.wav", "CRITICAL"),

    # 확인 / 터틀봇
    "EV_CHECK_REQUEST": ("tb_patrol_start.wav", "MEDIUM"),
    "EV_CONFIRM_DONE": ("tb_return.wav", "MEDIUM"),

    # 재료 요청 (임시로 detect.wav 사용)
    "EV_SALT": ("salt_move.wav", "MEDIUM"),
    "EV_PEPPER": ("pepper_move.wav", "MEDIUM"),
    "EV_SUGAR": ("suger_move.wav", "MEDIUM"),

    # 인식 실패
    "EV_UNKNOWN": ("error_voice.wav", "MEDIUM"),
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
# 핵심 단어 우선 매핑
# 우선순위:
# 1) 정지
# 2) 종료
# 3) 재료 요청
# 4) 시작
# 5) 확인 요청
# 6) 확인 완료
# =========================
def map_command(text: str):
    t = normalize_text(text)

    stop_tokens = [
        "멈춰",
        "그만",
        "정지",
        "멈추어",
    ]

    end_tokens = [
        "종료",
        "끝낼",
        "끝났",
        "끝",
        "다했",
        "다했어",
    ]

    salt_tokens = [
        "소금",
        "소금줘",
        "소금가져",
        "소금가져다줘",
    ]

    pepper_tokens = [
        "후추",
        "후추줘",
        "후추가져",
        "후추가져다줘",
    ]

    sugar_tokens = [
        "설탕",
        "설탕줘",
        "설탕가져",
        "설탕가져다줘",
    ]

    start_strong_tokens = [
        "시작",
    ]

    start_weak_tokens = [
        "조리",
        "요리",
    ]

    check_request_tokens = [
        "떨어",
        "도와",
        "찾아",
        "확인해",
        "뭐있",
        "뭐있어",
        "뭐있나",
    ]

    confirm_done_tokens = [
        "오케이",
        "알았",
        "확인됐",
        "됐다",
        "복귀",
        "돌아가",
        "복기",
    ]

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
        print("[오디오 에러] winsound를 사용할 수 없습니다. Windows 환경인지 확인하세요.")
        return

    winsound.PlaySound(str(file_path), winsound.SND_FILENAME)


# =========================
# 오디오 이벤트 등록
# =========================
def play_audio_event(event_code: str):
    if event_code not in EVENT_AUDIO_MAP:
        print(f"[오디오] 매핑 없음: {event_code}")
        return

    filename, priority_name = EVENT_AUDIO_MAP[event_code]
    priority = PRIORITY_MAP[priority_name]
    file_path = get_audio_path(filename)

    if file_path is None:
        print(f"[오디오] 파일 없음: {filename}")
        return

    audio_queue.put((priority, event_code, file_path))


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
                # 이미 더 높은 우선순위가 재생중이면 현재 이벤트는 무시
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
# Whisper 추론
# =========================
def transcribe_audio(model: WhisperModel, audio_data: np.ndarray):
    if audio_data is None or len(audio_data) == 0:
        print("[결과] 녹음된 데이터가 없음")
        play_audio_event("EV_UNKNOWN")
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
        play_audio_event("EV_UNKNOWN")
        return

    print(f"[원문] {text}")

    event_code = map_command(text)
    print(f"[이벤트] {event_code} ({event_description(event_code)})")

    play_audio_event(event_code)


# =========================
# 폴더/파일 체크
# =========================
def validate_audio_dir():
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"[오디오 폴더 없음] {AUDIO_DIR}")

    print(f"[오디오 폴더 확인] {AUDIO_DIR}")

    missing = []
    for _, (filename, _) in EVENT_AUDIO_MAP.items():
        if not (AUDIO_DIR / filename).exists():
            missing.append(filename)

    if missing:
        print("[주의] 아래 오디오 파일이 없습니다:")
        for m in sorted(set(missing)):
            print(f" - {m}")
    else:
        print("[오디오 파일 확인] 필수 파일 존재")


# =========================
# 메인
# =========================
def main():
    global is_recording, audio_chunks

    validate_audio_dir()

    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    audio_thread.start()

    print("[1] Whisper 모델 로딩 시작...")
    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )
    print("[2] Whisper 모델 로딩 완료")

    print("\n사용법:")
    print("- 스페이스바 누르고 있는 동안만 녹음")
    print("- 스페이스바 떼면 인식 후 음성 재생")
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
                    break

                current_space_state = keyboard.is_pressed("space")

                # 스페이스 눌렀을 때
                if current_space_state and not prev_space_state:
                    audio_chunks = []
                    is_recording = True
                    record_start_time = time.time()
                    print("[녹음] 시작...")

                # 스페이스 뗐을 때
                elif not current_space_state and prev_space_state:
                    is_recording = False
                    duration = time.time() - record_start_time if record_start_time else 0.0
                    print(f"[녹음] 종료 (길이: {duration:.2f}초)")

                    if duration < MIN_RECORD_SEC:
                        print("[경고] 너무 짧게 눌렀음. 다시 시도.\n")
                        play_audio_event("EV_UNKNOWN")
                    elif len(audio_chunks) == 0:
                        print("[경고] 녹음된 음성이 없음\n")
                        play_audio_event("EV_UNKNOWN")
                    else:
                        audio_data = np.concatenate(audio_chunks, axis=0).reshape(-1)
                        transcribe_audio(model, audio_data)
                        print()

                    record_start_time = None

                prev_space_state = current_space_state
                time.sleep(SLEEP_SEC)

            except KeyboardInterrupt:
                print("\n[종료] Ctrl+C 입력으로 종료합니다.")
                break
            except Exception as e:
                print(f"[에러] {e}")
                time.sleep(0.2)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[최상위 에러] {e}")
        input("엔터 누르면 종료")