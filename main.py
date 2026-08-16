import glob
import os
import site
import sys


def _add_nvidia_bins_to_path():
    # onnxruntime 只认 PATH（不认 os.add_dll_directory），把 CUDA/cuDNN 的 bin 加进 PATH。
    # 打包成 exe 后 getsitepackages 指向包内目录，所以要额外扫 _MEIPASS 和 exe 所在目录。
    roots = set()
    try:
        roots.update(site.getsitepackages())
    except Exception:
        pass
    roots.add(site.getusersitepackages())
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.add(meipass)
    if getattr(sys, "frozen", False):
        roots.add(os.path.dirname(sys.executable))

    bins = []
    for sp in roots:
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            bins.append(d)
    if meipass:
        bins.append(meipass)
    if bins:
        os.environ["PATH"] = os.pathsep.join(bins) + os.pathsep + os.environ["PATH"]


def _data_file_path():
    """face_points.json 的存放位置：优先 exe/脚本所在目录，不可写就退回用户主目录，
    避免打包后因工作目录不同而读写失败。"""
    candidates = []
    if getattr(sys, "frozen", False):
        candidates.append(os.path.join(os.path.dirname(sys.executable), "face_points.json"))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_points.json"))
    candidates.append(os.path.join(os.path.expanduser("~"), "face_points.json"))
    for p in candidates:
        d = os.path.dirname(p)
        if d and os.path.isdir(d) and os.access(d, os.W_OK):
            return p
    return candidates[-1]


_add_nvidia_bins_to_path()
DATA_FILE = _data_file_path()

from insightface.app import FaceAnalysis
import cv2
import numpy as np
import json
import time
from tkinter import simpledialog
import pyautogui
import pyperclip
import contextlib
import io

def normalize_points(points):
    """归一化：质心移到原点，再除以两眼外角距离（点36、45），消除位置和大小影响。"""
    center = points.mean(axis=0)
    p = points - center
    scale = np.linalg.norm(points[36] - points[45])
    if scale < 1e-6:
        scale = 1.0
    return p / scale


def align_to(reference, points):
    """把 points 旋转对齐到 reference（两者都已做过质心+尺度归一化），
    消除头部歪斜带来的差异，再比较才公平。"""
    try:
        H = reference.T @ points
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:  # 防止出现镜像
            Vt[-1, :] *= -1
            R = U @ Vt
        return points @ R.T
    except Exception:
        return points


def similarity(a, b):
    """先把 b 旋转对齐到 a，再算平均点距离 -> 相似度(0~1)，越大越像。"""
    b = align_to(a, b)
    d = np.linalg.norm(a - b, axis=1).mean()
    return 1.0 / (1.0 + d)


def parse_action(text):
    """把保存的 input 解析成动作 (mode, payload)。

    - 普通文字        -> ("paste", 文字)：触发时粘贴
    - "key:w"         -> ("tap", ["w"])：触发时按一下 w
    - "key:shift+w"   -> ("tap", ["shift", "w"])：组合键
    - "key:w hold"    -> ("hold", ["w"])：按住 w，直到表情不再匹配才松开

    支持中文键名：空格、回车、上、下、左、右。
    """
    text = (text or "").strip()
    if not text.lower().startswith("key:"):
        return ("paste", text)

    rest = text[4:].strip()
    hold = False
    parts = rest.split()
    if parts:
        rest = parts[0]
        if any(("hold" in p.lower()) or ("按住" in p) for p in parts[1:]):
            hold = True

    key_map = {
        "空格": "space", "space": "space",
        "回车": "enter", "enter": "enter",
        "上": "up", "up": "up", "下": "down", "down": "down",
        "左": "left", "left": "left", "右": "right", "right": "right",
        "shift": "shift", "ctrl": "ctrl", "alt": "alt",
        "esc": "esc", "tab": "tab", "capslock": "capslock",
    }
    keys = [key_map.get(k.strip().lower(), k.strip().lower()) for k in rest.split("+") if k.strip()]
    mode = "hold" if hold else "tap"
    return (mode, keys)


def load_originals():
    """读取 face_points.json，把每个表情解析成 (name, action, normalized_points)。"""
    originals = []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        for item in all_data:
            if "POINTS" in item:
                originals.append((item["name"], parse_action(item.get("input", "")),
                                  normalize_points(np.array(item["POINTS"], dtype=float))))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return originals


with contextlib.redirect_stdout(io.StringIO()):
    # 静音 insightface 加载模型时打印的日志
    app = FaceAnalysis(allowed_modules=['detection', 'landmark_3d_68'])
    app.prepare(ctx_id=-1, det_size=(640, 640))  # -1 = NVIDIA GPU

# 绕过 insightface 1.0.1 的 bug：prepare(ctx_id<0) 反而设成 CPU，这里改回 CUDA
for _m in app.models.values():
    _sess = getattr(_m, "session", None)
    if _sess is not None and hasattr(_sess, "set_providers"):
        try:
            _sess.set_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
        except Exception:
            pass

# 打包后 meanshape_68.pkl 定位会失效（insightface 用 __file__ 找数据文件），
# 它只用于 3D 姿态，本项目只要 68 点坐标，所以关掉 require_pose 跳过。
for _m in app.models.values():
    if getattr(_m, "require_pose", False):
        _m.require_pose = False

_gpu_ok = False
for _m in app.models.values():
    _sess = getattr(_m, "session", None)
    if _sess is not None and hasattr(_sess, "get_providers"):
        _gpu_ok = "CUDAExecutionProvider" in _sess.get_providers()
        break
print("已使用 GPU 加速" if _gpu_ok else "未使用 GPU（当前是 CPU 模式）")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows 上用 DSHOW 后端，比 MSMF 稳定

THRESHOLD = 0.9


def read_frame(cap):
    """读一帧。相机偶尔抽风读不到帧时重试几次，一直失败就重开摄像头，
    避免拿到空帧导致 imshow 崩溃。"""
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            return cap, frame
        cv2.waitKey(30)
    cap.release()
    new_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = new_cap.read()
    if ret and frame is not None:
        return new_cap, frame
    return new_cap, None


while True:
    cap, frame = read_frame(cap)
    if frame is None:
        continue
    key = cv2.waitKey(1) & 0xFF
    cv2.imshow("face", frame)
    if key == ord('q'):
        break

    if key == ord('x'):
        new_name = simpledialog.askstring("新建表情", "输入表情名字：")
        new_input = simpledialog.askstring("新建输入",
            "输入内容：\n普通文字=匹配时粘贴\nkey:键名=按一下键，如 key:w、key:空格\nkey:键名 hold=按住不放，如 key:w hold")
        if not new_name:
            continue
        faces = app.get(frame)
        if not faces:
            print("没有检测到人脸，请重新按 x")
            continue
        lm = faces[0].landmark_3d_68
        if lm is None:
            print("没有检测到3D关键点，请重新按 x")
            continue
        landmark68 = lm.astype(int).tolist()

        for j in range(68):
            cv2.circle(frame, (landmark68[j][0], landmark68[j][1]), 1, (255, 0, 0), thickness=-1)

        data = {"name": new_name,"input":new_input, "POINTS": landmark68}

        all_data = []
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, "r", encoding="utf-8") as f:
                    all_data = json.load(f)
            except json.JSONDecodeError:
                all_data = []

        # 同名覆盖，没有就追加
        found = False
        for i in range(len(all_data)):
            if all_data[i]["name"] == new_name:
                all_data[i] = data
                found = True
                break
        if not found:
            all_data.append(data)

        try:
            with open(DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=4)
            print("已保存到", DATA_FILE, "name =", new_name)
        except Exception as e:
            print("保存失败：", e)

        while True:
            cv2.imshow("face", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    if key == ord('z'):
        cv2.waitKey(100)   # 清掉残留按键，避免上次的 q 立刻退出本次 z
        was_below = True   # 从低于阈值跨到高于阈值才触发一次
        held_keys = []     # 当前被我们按住(hold)的键

        originals = load_originals()

        try:
            while True:
                cap, frame = read_frame(cap)
                if frame is None:
                    continue

                faces = app.get(frame, max_num=1)  # 只找 1 张脸
                if not faces:
                    cv2.imshow("face", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    continue
                lm = faces[0].landmark_3d_68
                if lm is None:
                    cv2.imshow("face", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    continue
                landmark68 = lm.astype(int).tolist()

                for j in range(68):
                    cv2.circle(frame, (landmark68[j][0], landmark68[j][1]), 1, (255, 0, 0), thickness=-1)

                # 遍历所有表情，取最相似的
                if originals:
                    new_pts = normalize_points(np.array(landmark68, dtype=float))
                    best_sim = 0.0
                    best_name = ""
                    best_action = ("paste", "")
                    for name, action, orig_pts in originals:
                        s = similarity(orig_pts, new_pts)
                        if s > best_sim:
                            best_sim = s
                            best_name = name
                            best_action = action

                    # 画面上实时显示相似度：绿=已到阈值，红=还没到
                    color = (0, 255, 0) if best_sim >= THRESHOLD else (0, 0, 255)
                    cv2.putText(frame, f"sim: {best_sim:.3f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    if best_sim >= THRESHOLD:
                        if was_below:
                            mode, payload = best_action
                            print("触发：", best_name, mode, payload, "相似度:", round(best_sim, 3))
                            if mode == "paste":
                                pyperclip.copy(payload)
                                pyautogui.hotkey("ctrl", "v")
                            elif mode == "tap":
                                if len(payload) > 1:
                                    pyautogui.hotkey(*payload)
                                else:
                                    pyautogui.press(payload[0])
                            elif mode == "hold":
                                for k in payload:
                                    if k not in held_keys:
                                        pyautogui.keyDown(k)
                                        held_keys.append(k)
                            was_below = False
                    else:
                        was_below = True
                        for k in held_keys:  # 表情不匹配了，松开所有 hold 键
                            pyautogui.keyUp(k)
                        held_keys.clear()

                cv2.imshow("face", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                time.sleep(0.05)  # 限制帧率：太卡就调大(如0.2)，想更流畅调小(如0.03)
        finally:
            for k in held_keys:  # 退出 z 模式时也保证松开
                pyautogui.keyUp(k)
            held_keys.clear()

cap.release()
cv2.destroyAllWindows()
