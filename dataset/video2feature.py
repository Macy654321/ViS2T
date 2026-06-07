import clip
import torch
import pandas as pd
import torch.nn as nn
import torchvision.transforms as T
import cv2
import os
import numpy as np
import random

def add_salt_pepper(img, amount=0.010, s_vs_p=0.5, seed=None):

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    out = img.copy()
    h, w = out.shape[:2]
    ch = 1 if out.ndim == 2 else out.shape[2]
    N = int(amount * h * w)  # Translated comment.

    # Salt
    num_salt = int(N * s_vs_p)
    ys = rng.integers(0, h, num_salt)
    xs = rng.integers(0, w, num_salt)
    if ch == 1:
        out[ys, xs] = 255
    else:
        out[ys, xs, :] = 255

    # Pepper
    num_pepper = N - num_salt
    ys = rng.integers(0, h, num_pepper)
    xs = rng.integers(0, w, num_pepper)
    if ch == 1:
        out[ys, xs] = 0
    else:
        out[ys, xs, :] = 0

    return out

def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    """
    Translated documentation.
      Translated documentation.
      Translated documentation.
      Translated documentation.
    """
    out = img.astype(np.float32) * float(alpha) + float(beta)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def motion_blur(img, ksize=9, angle=0):

    ksize = int(ksize)
    if ksize < 3: ksize = 3
    if ksize % 2 == 0: ksize += 1


    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0


    M = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))


    kernel_sum = kernel.sum()
    if kernel_sum != 0:
        kernel /= kernel_sum


    blurred = cv2.filter2D(img, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)
    return blurred
def random_occlusion_rgb(img, severity=3, fill="black"):

    severity = int(np.clip(severity, 1, 5))
    h, w = img.shape[:2]
    out = img.copy()


    holes = {1:(1,0.10), 2:(1,0.15), 3:(5,0.1), 4:(3,0.25), 5:(4,0.30)}
    num_holes, max_frac = holes[severity]
    max_h = int(h * max_frac)
    max_w = int(w * max_frac)


    def pick_color():
        if isinstance(fill, tuple) and len(fill) == 3:
            return tuple(int(c) for c in fill)
        if fill == "white":  return (255, 255, 255)
        if fill == "random": return (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        return (0, 0, 0)  # black

    for _ in range(num_holes):

        rect_h = random.randint(max(4, max_h // 4), max_h) if max_h > 4 else max_h or 1
        rect_w = random.randint(max(4, max_w // 4), max_w) if max_w > 4 else max_w or 1


        y1 = random.randint(0, max(0, h - rect_h))
        x1 = random.randint(0, max(0, w - rect_w))
        y2, x2 = y1 + rect_h, x1 + rect_w

        color = pick_color()
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=-1)

    return out
def extract_and_save_features(video_path, model, save_dir, max_frames=30):

    model = model.eval()

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(save_dir, f"{video_name}.pt")

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamps = np.linspace(0, total_frames-1, max_frames, dtype=int)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    features = []

    for fn in timestamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)  # Translated comment.
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img1 = random_occlusion_rgb(img)
        img2 = motion_blur(img,7)
        img3 = adjust_brightness_contrast(img, alpha=1.2, beta=35)
        img4 = adjust_brightness_contrast(img, alpha=0.9, beta=-15)
        img5 = add_salt_pepper(img)
        cv2.imwrite("image.jpg", img)
        cv2.imwrite("blur.jpg", img2)
        cv2.imwrite("brignt.jpg", img3)
        cv2.imwrite("dark.jpg", img4)
        cv2.imwrite("salt.jpg", img5)
        exit()
        tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

        with torch.no_grad():
            feat = model.encode_image(tensor)  # Translated comment.
            features.append(feat.squeeze(0))  # Translated comment.

    cap.release()


    if features:
        result = torch.stack(features)  # [N, 512]
        torch.save(result, save_path)
        print(f"Saved features: {save_path}, shape = {result.shape}")
    else:
        print(f"No frames were extracted successfully: {video_path}")

# df = pd.read_csv('panda/train_data.csv',header=None)
# data_list=[]
# for i in range(df.shape[0]):
#     data_list.append(df.iloc[i,0])

def get_video_names(folder, exts=(".mp4", ".avi", ".mkv", ".mov")):
    video_list = [
        f for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ]
    return video_list

# Translated comment.
# folder = "panda/videos"
# data_list = get_video_names(folder)

df = pd.read_csv("panda/test_data.csv", header=None)

data_list = df.iloc[:, 0].tolist()
base_dir="panda/videos"
save_dir="panda/salt"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load('ViT-B/32', device=device,download_root="../pretrained/vit")
extract_and_save_features("panda/videos/D000001.mp4", model, save_dir, max_frames=30)
# exit()
# extract_and_save_features("msvd/videos/0bSz70pYAP0_5_15.avi", model, save_dir, max_frames=8)
for i in range(0,len(data_list)):
    video_path =os.path.join(base_dir,data_list[i]+".mp4")
    extract_and_save_features(video_path, model, save_dir, max_frames=30)
    # visual_feature = model.encode_image(image)
