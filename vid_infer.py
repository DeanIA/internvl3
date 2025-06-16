import os
# Only expose GPU #1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
import torch
import numpy as np
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# ─── Model Setup 78B 8 bit quant ─────────────────────────────────────────────────
MODEL_PATH = "OpenGVLab/InternVL3-78B"
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# ─── Constants ───────────────────────────────────────────────────
INPUT_SIZE = 448
NUM_FRAMES = 16
GEN_CONFIG = dict(max_new_tokens=1024, do_sample=True)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# ─── Transforms ─────────────────────────────────────────────────
def build_transform():
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB')),
        T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

# ─── Load and Preprocess Video ──────────────────────────────────
def get_frame_indices(num_frames, total):
    return np.linspace(0, total - 1, num_frames, dtype=int)

def load_video(video_path, num_frames):
    vr = VideoReader(video_path, ctx=cpu(0))
    transform = build_transform()
    indices = get_frame_indices(num_frames, len(vr))
    pixel_values = [transform(Image.fromarray(vr[i].asnumpy())) for i in indices]
    return torch.stack(pixel_values)  # [num_frames, 3, H, W]

# ─── Inference ──────────────────────────────────────────────────
def infer(video_path):
    video_tensor = load_video(video_path, NUM_FRAMES).to(torch.bfloat16).cuda()
    prompt = ''.join([f'Frame{i+1}: <image>\n' for i in range(NUM_FRAMES)])
    prompt += "Provide a detailed description of each image. Describe the foreground and background separately. Mention any people, objects, and actions clearly. What are the people doing? What expressions or activities are visible? What is the setting or context? Is there violence happening?"

    response, _ = model.chat(
        tokenizer,
        video_tensor,
        prompt,
        GEN_CONFIG,
        history=None,
        return_history=True
    )
    print("User:", prompt)
    print("Assistant:", response)

# ─── Entry ──────────────────────────────────────────────────────
if __name__ == "__main__":
    infer("./TNS_0119_V.mp4")