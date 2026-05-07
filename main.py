"""posture-guard: 摄像头颈椎前倾监测。"""

import os
import subprocess
import sys
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

# ---------- 配置 ----------
SAMPLE_INTERVAL = 10.0
# 长 interval 时把 sleep 拆短：防止 macOS 挂起长 sleep（休眠 / App Nap），并让进程周期性"心跳"。
SLEEP_CHUNK = 60.0
# 离座长间隔下，每 IDLE_POLL_CHUNK 秒查一次键鼠 idle。
IDLE_POLL_CHUNK = 30.0

# YOLO11n-pose 是真正的"先检测人 → 再标关键点"两阶段联合模型，没人时不输出 detection，
# 从根上消除 BlazePose 系在空椅子上幻觉关键点的问题。模型经 ONNX 导出后用 onnxruntime 直推，
# 完全不依赖 ultralytics/torch，运行时常驻内存约 100–200MB。
MODEL_PATH = "yolo11n-pose.onnx"
MODEL_INPUT_SIZE = 640
PERSON_CONF = 0.5    # YOLO 人物 detection 置信度阈值
KEYPOINT_CONF = 0.5  # 单关键点 visibility 阈值

ALERT_SOUND_PATH = "alert.wav"

# 头肩高度比阈值：(肩 y - 头 y) / 肩宽。端正 ~0.7+，低头时减小
LEAN_RATIO = 0.5     # 低于此 → 轻度低头
DEEP_RATIO = 0.3     # 低于此 → 严重低头
EMA_ALPHA = 0.5      # 时序平滑系数（新值权重）

# YOLO 已根治幻觉，下面这些只用来判断"头肩比能不能算"——算不出就让 head_ratio=None，
# has_person 仍由 YOLO 的 person_conf 单独决定（避免低头吃东西时头部点 visibility
# 偏低被误判离座）。
MIN_HEAD_POINTS = 2              # 头部 5 点中至少 N 个可见才能取头部均值
MIN_SHOULDER_WIDTH_PX = 30       # 肩宽过窄会让 ratio 数值不稳
RATIO_VALID_RANGE = (-2.0, 2.0)  # 头肩比物理合理区间

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

# COCO 17 关键点索引（YOLO-pose 标准布局）
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
HEAD_POINTS = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)


MODEL_EXPORT_HINT = """
缺少模型文件 {path}。一次性导出（uvx 临时托管 ultralytics+torch，导出完即可弃用）：

  uvx --with onnx --with onnxslim --from ultralytics yolo export model=yolo11n-pose.pt format=onnx imgsz={size}

执行完会在当前目录生成 yolo11n-pose.onnx (~11 MB)。
"""


# ---------- 工具 ----------
def fmt_dur(sec):
    """中文人读时长。<60s 显示秒；<1h 分（带零头秒）；≥1h 时分。"""
    sec = max(0, int(sec))
    if sec < 60:
        return f"{sec}秒"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m}分{s:02d}秒" if s else f"{m}分钟"
    h, m = divmod(m, 60)
    return f"{h}时{m:02d}分" if m else f"{h}小时"


# ---------- 模型推理 ----------
def detect_keypoints(session, frame):
    """跑 YOLO11n-pose，返回最高置信度人物的 17 关键点 (numpy [17, 3] = x_px, y_px, conf)。
    画面无人时返回 None——这是 YOLO 相对 BlazePose 的本质改进：先做人物检测，没人就是没人。"""
    h, w = frame.shape[:2]
    # letterbox：等比缩放 + 右/下灰边 (114)，保持人体不被拉伸
    scale = MODEL_INPUT_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    # HWC uint8 → NCHW float32 ∈ [0, 1]
    tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    # 输出 shape (1, 56, num_anchors)。56 = 4 (cx,cy,w,h) + 1 (person_conf) + 17*3 (kp_x, kp_y, kp_conf)
    pred = output[0].T  # → (num_anchors, 56)
    conf = pred[:, 4]
    if conf.max() < PERSON_CONF:
        return None
    best = pred[int(np.argmax(conf))]
    kp = best[5:].reshape(17, 3).astype(np.float32).copy()
    # 关键点从 letterbox 输入坐标系映回原图像素
    kp[:, 0] /= scale
    kp[:, 1] /= scale
    return kp


# ---------- 单帧分析 ----------
def analyze_pose(kp):
    """kp: (17, 3) numpy 数组 [x, y, conf]，None 表示 YOLO 没检出人。
    YOLO 的 person_conf 已经在 detect_keypoints 里挡过阈值，能拿到 kp 就是有人。
    head_ratio 只在关键点齐全时才算，缺点就让 ratio=None 让 EMA/窗口跳过这一帧
    （状态机已支持），避免低头吃东西/喝水时被误判离座。"""
    if kp is None:
        return {"has_person": False, "head_ratio": None}

    visible = lambda i: kp[i, 2] >= KEYPOINT_CONF

    # 双肩任一不可见 → 算不了肩宽，跳过这一帧的 ratio
    if min(kp[L_SHOULDER, 2], kp[R_SHOULDER, 2]) < KEYPOINT_CONF:
        return {"has_person": True, "head_ratio": None}

    head_visible = [i for i in HEAD_POINTS if visible(i)]
    if len(head_visible) < MIN_HEAD_POINTS:
        return {"has_person": True, "head_ratio": None}

    l_sh, r_sh = kp[L_SHOULDER, :2], kp[R_SHOULDER, :2]
    sh_w = abs(l_sh[0] - r_sh[0])
    if sh_w < MIN_SHOULDER_WIDTH_PX:
        return {"has_person": True, "head_ratio": None}

    sh_mid_y = (l_sh[1] + r_sh[1]) / 2
    head_y = sum(kp[i, 1] for i in head_visible) / len(head_visible)

    # 头比肩高出多少倍肩宽。端正 ~0.7+，低头减小，趴桌为负
    ratio = (sh_mid_y - head_y) / sh_w
    if not (RATIO_VALID_RANGE[0] <= ratio <= RATIO_VALID_RANGE[1]):
        return {"has_person": True, "head_ratio": None}
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
    if not os.path.exists(MODEL_PATH):
        print(MODEL_EXPORT_HINT.format(path=MODEL_PATH, size=MODEL_INPUT_SIZE))
        sys.exit(1)

    print(f"[posture-guard] 加载 {MODEL_PATH}…")
    # macOS arm64 优先 CoreML EP（自动 fallback CPU），其余平台直接 CPU
    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"] if sys.platform == "darwin" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(MODEL_PATH, providers=providers)

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
            kp = detect_keypoints(session, frame)
            infer_ms = (time.time() - infer_start) * 1000

            obs = analyze_pose(kp)
            now = time.time()
            # update_state 会清 leave_ts 并可能重置 sitting_total_sec，所以提前快照
            was_present = state["is_present"]
            prev_leave_ts = state["leave_ts"]
            prev_total = state["sitting_total_sec"]
            for a in update_state(state, obs, now):
                title, msg = ALERT_MSG[a]
                notify(title, msg)
                print(f"[ALERT] {title}: {msg}")

            # 刚从离座切回在座：只在状态切换那一帧打一次回归提示，不刷屏
            if not was_present and state["is_present"] and prev_leave_ts > 0:
                rest_sec = now - prev_leave_ts
                ts = time.strftime('%H:%M:%S')
                if rest_sec >= ACTIVITY_RESET_SEC:
                    print(f"[{ts}] 欢迎回来，休息了 {fmt_dur(rest_sec)}，坐姿计时已重置")
                elif prev_total > 0:
                    need = ACTIVITY_RESET_SEC - rest_sec
                    print(
                        f"[{ts}] 欢迎回来，休息 {fmt_dur(rest_sec)}（未满 {fmt_dur(ACTIVITY_RESET_SEC)}，"
                        f"还差 {fmt_dur(need)}才会重置），坐姿计时延续 {fmt_dur(prev_total)}"
                    )

            absent_sec = now - state["last_seen_ts"]
            if absent_sec >= ABSENT_LONG_SEC:
                interval = INTERVAL_LONG
            elif absent_sec >= ABSENT_MEDIUM_SEC:
                interval = INTERVAL_MEDIUM
            else:
                interval = SAMPLE_INTERVAL

            tags = []
            if obs["has_person"]:
                sat = state["sitting_total_sec"]
                remaining = SITTING_LIMIT_SEC - sat
                if remaining > 0:
                    tags.append(f"在座 {fmt_dur(sat)} (距提醒 {fmt_dur(remaining)})")
                else:
                    tags.append(f"在座 {fmt_dur(sat)} (已超 {fmt_dur(-remaining)})")
                raw = obs["head_ratio"]
                ema = state["ema_ratio"]
                if raw is not None:
                    tags.append(f"头肩比 {raw:.2f}/平滑 {ema:.2f}")
                    if ema < DEEP_RATIO:
                        tags.append("严重低头!")
                    elif ema < LEAN_RATIO:
                        tags.append("前倾")
            else:
                # 离座中：之前若有累积坐姿，提示是否已休息够（≥10 分钟回来即重置）
                rest_tag = f"离座 {fmt_dur(absent_sec)}"
                if state["sitting_total_sec"] > 0:
                    if absent_sec < ACTIVITY_RESET_SEC:
                        rest_tag += f" (再休息 {fmt_dur(ACTIVITY_RESET_SEC - absent_sec)} 可重置坐姿 {fmt_dur(state['sitting_total_sec'])})"
                    else:
                        rest_tag += " (已休息够，回来重新计时)"
                tags.append(f"{rest_tag} 下次 {fmt_dur(interval)}")
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


if __name__ == "__main__":
    main()
