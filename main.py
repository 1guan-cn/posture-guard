"""posture-guard: 摄像头颈椎前倾监测，每 3 秒采样一次。"""

import subprocess
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# ---------- 配置 ----------
SAMPLE_INTERVAL = 3.0
MODEL_PATH = "yolov8n-pose.pt"
KP_CONF = 0.5

# 头肩高度比阈值：(肩 y - 头 y) / 肩宽。端正 ~0.7+，低头时减小
LEAN_RATIO = 0.5     # 低于此 → 轻度低头
DEEP_RATIO = 0.3     # 低于此 → 严重低头
EMA_ALPHA = 0.4      # 时序平滑系数（新值权重）

LEAN_WINDOW = 20     # 60 秒（20 帧 × 3 秒）
DEEP_WINDOW = 10     # 30 秒
HIT_RATIO = 0.7

ABSENT_FRAMES_TO_LEAVE = 2     # 连续 ≥6 秒未检出 = 离座
SITTING_LIMIT_SEC = 40 * 60
ACTIVITY_RESET_SEC = 10 * 60   # 离座 ≥10 分钟重置久坐
ALERT_COOLDOWN_SEC = 5 * 60

# COCO 17 关键点索引
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
HEAD_POINTS = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)


# ---------- 单帧分析 ----------
def analyze_pose(kp_xy, kp_conf):
    if kp_xy is None or len(kp_xy) == 0:
        return {"has_person": False, "head_ratio": None}

    visible = lambda i: kp_conf[i] >= KP_CONF

    # 双肩是几何判定的最低要求
    if not (visible(L_SHOULDER) and visible(R_SHOULDER)):
        return {"has_person": False, "head_ratio": None}

    l_sh, r_sh = kp_xy[L_SHOULDER], kp_xy[R_SHOULDER]
    sh_mid_y = (l_sh[1] + r_sh[1]) / 2
    sh_w = abs(l_sh[0] - r_sh[0]) or 1.0

    # 用所有可见头部点（鼻+双眼+双耳）的 y 取平均，比单点稳得多
    head_ys = [kp_xy[i][1] for i in HEAD_POINTS if visible(i)]
    if not head_ys:
        return {"has_person": True, "head_ratio": None}
    head_y = sum(head_ys) / len(head_ys)

    # 头比肩高出多少倍肩宽。端正 ~0.7+，低头减小，趴桌为负
    ratio = (sh_mid_y - head_y) / sh_w
    return {"has_person": True, "head_ratio": float(ratio)}


# ---------- 状态机 ----------
def new_state():
    return {
        "sitting_start_ts": 0.0,
        "sitting_total_sec": 0.0,
        "last_seen_ts": 0.0,
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
        # 重度优先于轻度
        if len(s["deep_window"]) >= DEEP_WINDOW:
            if sum(s["deep_window"]) / DEEP_WINDOW > HIT_RATIO and can_alert("deep"):
                alerts.append("deep")
                s["last_alert_ts"]["deep"] = now
        if len(s["lean_window"]) >= LEAN_WINDOW:
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
    # 异步播 Hero 音 + 异步弹常驻对话框（10 分钟没操作自动关闭，防止挂死）
    subprocess.Popen(["afplay", "/System/Library/Sounds/Hero.aiff"])
    script = (
        f'display dialog "{safe_msg}" with title "{safe_title}" '
        f'buttons {{"知道了"}} default button "知道了" '
        f'with icon caution giving up after 600'
    )
    subprocess.Popen(["osascript", "-e", script])


# ---------- 主循环 ----------
def main():
    print(f"[posture-guard] 加载模型 {MODEL_PATH}…")
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头（系统设置 → 隐私与安全性 → 摄像头 中授予终端权限）")

    state = new_state()
    print(f"[posture-guard] 启动完成，每 {SAMPLE_INTERVAL}s 采样一次，Ctrl+C 退出")

    try:
        while True:
            tick = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(SAMPLE_INTERVAL)
                continue

            results = model(frame, verbose=False)
            kp_xy, kp_conf = None, None
            if results and results[0].keypoints is not None:
                kps = results[0].keypoints
                if kps.xy is not None and len(kps.xy) > 0:
                    kp_xy = kps.xy[0].cpu().numpy()
                    kp_conf = (
                        kps.conf[0].cpu().numpy()
                        if kps.conf is not None
                        else np.ones(len(kp_xy))
                    )

            obs = analyze_pose(kp_xy, kp_conf)
            now = time.time()
            for a in update_state(state, obs, now):
                title, msg = ALERT_MSG[a]
                notify(title, msg)
                print(f"[ALERT] {title}: {msg}")

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
                tags.append("离座")
            print(f"[{time.strftime('%H:%M:%S')}] {' '.join(tags)}")

            elapsed = time.time() - tick
            time.sleep(max(0.0, SAMPLE_INTERVAL - elapsed))
    except KeyboardInterrupt:
        print("\n[posture-guard] 已退出")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
