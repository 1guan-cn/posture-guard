"""posture-guard: 摄像头颈椎前倾监测。"""

import os
import threading


def _silence_mediapipe_logs():
    """MediaPipe C++ 层直接写 fd 2，Python 层环境变量（GLOG_minloglevel）管不住。
    用 fd 重定向 + 后台线程过滤的方式拦截噪音日志（含 Google clearcut 遥测尝试上传的 ERROR）。"""
    noise = (
        "clearcut",
        "Source Location Trace",
        "wireless/android",
        "inference_feedback_manager",
        "init-domain",
        "gl_context.cc",
        "TensorFlow Lite XNNPACK",
        "landmark_projection_calculator",
    )
    saved_fd = os.dup(2)
    r, w = os.pipe()
    os.dup2(w, 2)
    os.close(w)

    def pump():
        buf = b""
        while True:
            chunk = os.read(r, 4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not any(p in line.decode("utf-8", "replace") for p in noise):
                    os.write(saved_fd, line + b"\n")

    threading.Thread(target=pump, daemon=True).start()


_silence_mediapipe_logs()

import subprocess
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------- 配置 ----------
SAMPLE_INTERVAL = 10.0
# 长 interval 时把 sleep 拆短：防止 macOS 挂起长 sleep（休眠 / App Nap），并让进程周期性"心跳"。
# 真正开摄像头 + 推理仍按 interval 走，所以省电效果不变。
SLEEP_CHUNK = 60.0
# 离座长间隔下，每 IDLE_POLL_CHUNK 秒查一次键鼠 idle。
# 判定：idle 比这一段 sleep 实际时长还短 → sleep 期间 idle 被重置过 → 必定发生过键鼠活动。
# 不能用固定阈值（如 idle<5s），那只能捕获"查询瞬间正好在动"，会漏掉 sleep 中段动一下的场景。
IDLE_POLL_CHUNK = 30.0
KP_CONF = 0.5      # 单点 visibility "可见"阈值（用于头部均值参与判定）
ANCHOR_CONF = 0.7  # 关键 anchor 点（双肩、头部至少 1 个）"高置信度"阈值，挡幻觉人
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
POSE_MODEL_PATH = "pose_landmarker_lite.task"
ALERT_SOUND_PATH = "alert.wav"  # 项目根目录的本地音效

# 头肩高度比阈值：(肩 y - 头 y) / 肩宽。端正 ~0.7+，低头时减小
LEAN_RATIO = 0.5     # 低于此 → 轻度低头
DEEP_RATIO = 0.3     # 低于此 → 严重低头
EMA_ALPHA = 0.5      # 时序平滑系数（新值权重）

# 抗椅子误识别：BlazePose 在空椅子上会给虚假关键点（典型症状是 sh_w≈0 让头肩比爆炸到几百几千）
MIN_HEAD_POINTS = 2              # 头部 5 点中至少 N 个可见才算人——椅子上不会有头
MIN_SHOULDER_WIDTH_PX = 30       # 肩宽过窄即噪声
RATIO_VALID_RANGE = (-2.0, 2.0)  # 头肩比物理合理区间，超出即丢弃

LEAN_WINDOW = 6      # 60 秒（6 帧 × 10 秒）
DEEP_WINDOW = 6      # 60 秒
HIT_RATIO = 0.7

ABSENT_FRAMES_TO_LEAVE = 2     # 连续 ≥20 秒未检出 = 离座
SITTING_LIMIT_SEC = 40 * 60
ACTIVITY_RESET_SEC = 10 * 60   # 离座 ≥10 分钟重置久坐
ALERT_COOLDOWN_SEC = 5 * 60

# 离座自适应采样：长时间没人就拉长间隔，省 CPU 和摄像头闪灯
ABSENT_MEDIUM_SEC = 5 * 60     # 离座 ≥5 分钟 → 间隔变 10 分钟
ABSENT_LONG_SEC = 30 * 60      # 离座 ≥30 分钟 → 间隔变 30 分钟
INTERVAL_MEDIUM = 10 * 60.0
INTERVAL_LONG = 30 * 60.0

# MediaPipe BlazePose 33 关键点索引
NOSE = 0
L_EYE, R_EYE = 2, 5
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
HEAD_POINTS = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)


# ---------- 单帧分析 ----------
def analyze_pose(kp_xy, kp_conf):
    if kp_xy is None or len(kp_xy) == 0:
        return {"has_person": False, "head_ratio": None}

    visible = lambda i: kp_conf[i] >= KP_CONF

    # 双肩必须高置信度——挡 BlazePose 在空画面时"幻觉"出形态合理的虚假人
    if min(kp_conf[L_SHOULDER], kp_conf[R_SHOULDER]) < ANCHOR_CONF:
        return {"has_person": False, "head_ratio": None}

    # 头部至少 N 个点可见，且至少 1 个高置信度——椅子/背景上没头
    head_visible = [i for i in HEAD_POINTS if visible(i)]
    if len(head_visible) < MIN_HEAD_POINTS:
        return {"has_person": False, "head_ratio": None}
    if max(kp_conf[i] for i in head_visible) < ANCHOR_CONF:
        return {"has_person": False, "head_ratio": None}

    l_sh, r_sh = kp_xy[L_SHOULDER], kp_xy[R_SHOULDER]
    sh_w = abs(l_sh[0] - r_sh[0])
    # 肩宽过窄说明检测失真（典型空椅子 sh_w≈0 会让 ratio 爆到几百几千）
    if sh_w < MIN_SHOULDER_WIDTH_PX:
        return {"has_person": False, "head_ratio": None}

    sh_mid_y = (l_sh[1] + r_sh[1]) / 2
    head_y = sum(kp_xy[i][1] for i in head_visible) / len(head_visible)

    # 头比肩高出多少倍肩宽。端正 ~0.7+，低头减小，趴桌为负
    ratio = (sh_mid_y - head_y) / sh_w
    if not (RATIO_VALID_RANGE[0] <= ratio <= RATIO_VALID_RANGE[1]):
        return {"has_person": False, "head_ratio": None}
    return {"has_person": True, "head_ratio": float(ratio)}


# ---------- 状态机 ----------
def new_state(now=None):
    now = now if now is not None else time.time()
    return {
        "sitting_start_ts": 0.0,
        "sitting_total_sec": 0.0,
        "last_seen_ts": now,  # 启动时给个非 0 值，避免 absent_sec 一上来就巨大、立刻跳进长间隔
        "absent_frames": 0,
        "leave_ts": 0.0,
        "is_present": False,
        "ema_ratio": None,    # EMA 平滑后的头肩比
        "lean_window": deque(maxlen=LEAN_WINDOW),
        "deep_window": deque(maxlen=DEEP_WINDOW),
        "last_alert_ts": {"sit": 0.0, "lean": 0.0, "deep": 0.0},
    }


def update_state(s, obs, now):
    alerts = []

    if obs["has_person"]:
        s["absent_frames"] = 0
        if not s["is_present"]:
            if s["leave_ts"] > 0:
                if now - s["leave_ts"] >= ACTIVITY_RESET_SEC:
                    s["sitting_total_sec"] = 0.0
                s["leave_ts"] = 0.0
            s["is_present"] = True
            s["sitting_start_ts"] = now - s["sitting_total_sec"]
        s["sitting_total_sec"] = now - s["sitting_start_ts"]
        s["last_seen_ts"] = now
        r = obs["head_ratio"]
        if r is not None:
            s["ema_ratio"] = r if s["ema_ratio"] is None else EMA_ALPHA * r + (1 - EMA_ALPHA) * s["ema_ratio"]
            smooth = s["ema_ratio"]
            s["lean_window"].append(1 if smooth < LEAN_RATIO else 0)
            s["deep_window"].append(1 if smooth < DEEP_RATIO else 0)
    else:
        s["absent_frames"] += 1
        if s["is_present"] and s["absent_frames"] >= ABSENT_FRAMES_TO_LEAVE:
            s["is_present"] = False
            s["leave_ts"] = s["last_seen_ts"]
            s["ema_ratio"] = None
            s["lean_window"].clear()
            s["deep_window"].clear()

    def can_alert(k):
        return now - s["last_alert_ts"][k] >= ALERT_COOLDOWN_SEC

    if s["is_present"]:
        if s["sitting_total_sec"] >= SITTING_LIMIT_SEC and can_alert("sit"):
            alerts.append("sit")
            s["last_alert_ts"]["sit"] = now
        # 重度优先：触发 deep 时同时 mute lean，避免低头看手机时收到两条提醒
        deep_fired = False
        if len(s["deep_window"]) >= DEEP_WINDOW:
            if sum(s["deep_window"]) / DEEP_WINDOW > HIT_RATIO and can_alert("deep"):
                alerts.append("deep")
                s["last_alert_ts"]["deep"] = now
                s["last_alert_ts"]["lean"] = now
                deep_fired = True
        if not deep_fired and len(s["lean_window"]) >= LEAN_WINDOW:
            if sum(s["lean_window"]) / LEAN_WINDOW > HIT_RATIO and can_alert("lean"):
                alerts.append("lean")
                s["last_alert_ts"]["lean"] = now

    return alerts


# ---------- 通知 ----------
ALERT_MSG = {
    "sit":  ("该起来活动了", "已经连续坐 40 分钟，起来走走吧"),
    "lean": ("调整坐姿",     "颈部前倾过久，挺直背"),
    "deep": ("抬头",         "颈椎深度前屈，可能在低头看手机或键盘"),
}


def notify(title, msg):
    safe_title = title.replace('"', '\\"')
    safe_msg = msg.replace('"', '\\"')
    # 异步播本地音效 + 异步弹常驻对话框（10 分钟没操作自动关闭，防止挂死）。
    # stdout/stderr 重定向到 DEVNULL：避免 dialog 回执（"button returned:知道了"）污染主日志
    if os.path.exists(ALERT_SOUND_PATH):
        subprocess.Popen(
            ["afplay", ALERT_SOUND_PATH],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        print(f"[posture-guard] 警告: 音效文件 {ALERT_SOUND_PATH} 不存在，跳过播放")
    script = (
        f'display dialog "{safe_msg}" with title "{safe_title}" '
        f'buttons {{"知道了"}} default button "知道了" '
        f'with icon caution giving up after 600'
    )
    subprocess.Popen(
        ["osascript", "-e", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------- 主循环 ----------
def ensure_model():
    if not os.path.exists(POSE_MODEL_PATH):
        print(f"[posture-guard] 下载 MediaPipe Pose 模型 (~6 MB)…")
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)


def hid_idle_sec():
    """读 macOS IOHIDSystem 的键鼠 idle 秒数（触控板/鼠标/键盘共用同一计数器）。读取失败返回 None。"""
    try:
        out = subprocess.check_output(
            ["ioreg", "-c", "IOHIDSystem"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode("utf-8", "replace")
    except Exception:
        return None
    for line in out.splitlines():
        if "HIDIdleTime" in line:
            try:
                return int(line.rsplit("=", 1)[1].strip()) / 1e9
            except ValueError:
                return None
    return None


def grab_frame():
    """按需开关摄像头：sleep 期间释放设备，避免 OpenCV 后台管线吃 CPU。代价：摄像头指示灯每次采样会闪一下。"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # 摄像头需要预热几帧才能得到稳定曝光帧
    for _ in range(3):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main():
    ensure_model()
    print("[posture-guard] 加载 MediaPipe Pose (lite)…")
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        # 三个阈值用 ANCHOR_CONF：MP 内部检测/存在/跟踪环节就先过滤掉低置信度的虚假姿态
        min_pose_detection_confidence=ANCHOR_CONF,
        min_pose_presence_confidence=ANCHOR_CONF,
        min_tracking_confidence=ANCHOR_CONF,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    state = new_state()
    print(
        f"[posture-guard] 启动完成，在座/短暂离座每 {SAMPLE_INTERVAL:.0f}s 采样，"
        f"离座 ≥{ABSENT_MEDIUM_SEC // 60} 分钟 → {INTERVAL_MEDIUM / 60:.0f} 分钟，"
        f"≥{ABSENT_LONG_SEC // 60} 分钟 → {INTERVAL_LONG / 60:.0f} 分钟。Ctrl+C 退出"
    )

    try:
        while True:
            tick = time.time()
            frame = grab_frame()
            if frame is None:
                # 摄像头被其他进程占用（Zoom/腾讯会议等）时的静默期，必须打日志，否则用户看不出区别于"卡死"
                print(f"[{time.strftime('%H:%M:%S')}] 摄像头不可用，{int(SAMPLE_INTERVAL)}s 后重试")
                time.sleep(SAMPLE_INTERVAL)
                continue

            infer_start = time.time()
            # MediaPipe 要 RGB；归一化坐标乘回像素，让 x/y 单位一致
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)
            infer_ms = (time.time() - infer_start) * 1000
            kp_xy, kp_conf = None, None
            if result.pose_landmarks:
                h, w = frame.shape[:2]
                lms = result.pose_landmarks[0]
                kp_xy = np.array([(lm.x * w, lm.y * h) for lm in lms])
                kp_conf = np.array([lm.visibility for lm in lms])

            obs = analyze_pose(kp_xy, kp_conf)
            now = time.time()
            for a in update_state(state, obs, now):
                title, msg = ALERT_MSG[a]
                notify(title, msg)
                print(f"[ALERT] {title}: {msg}")

            absent_sec = now - state["last_seen_ts"]
            if absent_sec >= ABSENT_LONG_SEC:
                interval = INTERVAL_LONG
            elif absent_sec >= ABSENT_MEDIUM_SEC:
                interval = INTERVAL_MEDIUM
            else:
                interval = SAMPLE_INTERVAL

            tags = []
            if obs["has_person"]:
                tags.append(f"在座 {int(state['sitting_total_sec'])}s")
                raw = obs["head_ratio"]
                ema = state["ema_ratio"]
                if raw is not None:
                    tags.append(f"头肩比 {raw:.2f}/平滑 {ema:.2f}")
                    if ema < DEEP_RATIO:
                        tags.append("严重低头!")
                    elif ema < LEAN_RATIO:
                        tags.append("前倾")
            else:
                tags.append(f"离座 {int(absent_sec)}s 下次 {int(interval)}s")
            print(f"[{time.strftime('%H:%M:%S')}] {' '.join(tags)} | 推理 {infer_ms:.0f}ms")

            # 拆短 sleep：长 sleep 在 macOS 上会被休眠/挂起冻结，醒来后要等剩余时间走完，导致采样"卡死"
            # 离座长间隔时改用 IDLE_POLL_CHUNK 步长，每步查一次键鼠 idle → 用户回来动鼠标键盘可被快速捕获
            deadline = tick + interval
            use_hid_wake = interval > SAMPLE_INTERVAL
            chunk = IDLE_POLL_CHUNK if use_hid_wake else SLEEP_CHUNK
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                sleep_dur = min(chunk, remaining)
                time.sleep(sleep_dur)
                if use_hid_wake:
                    idle = hid_idle_sec()
                    # +1s 容差：ioreg 调用本身和 sleep 唤醒抖动各占几十~几百毫秒
                    if idle is not None and idle < sleep_dur + 1.0:
                        print(f"[{time.strftime('%H:%M:%S')}] 检测到键鼠活动 (idle {idle:.1f}s, sleep {sleep_dur:.0f}s)，提前重采样")
                        break
    except KeyboardInterrupt:
        print("\n[posture-guard] 已退出")
    finally:
        landmarker.close()


if __name__ == "__main__":
    main()
