import os
# Only expose GPU #1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig

# ─── Configuration ──────────────────────────────────────────────────────────
MODEL_PATH     = "OpenGVLab/InternVL3-1B"
PRM_PATH       = "OpenGVLab/VisualPRM-8B-v1_1"
INPUT_SIZE     = 448
NUM_FRAMES     = 16
MAX_TILES      = 12
GEN_CONFIG     = dict(max_new_tokens=1024, do_sample=True)
IMAGENET_MEAN  = (0.485, 0.456, 0.406)
IMAGENET_STD   = (0.229, 0.224, 0.225)
MAX_CONTEXT    = 12288

# ─── Load Models ─────────────────────────────────────────────────────────────
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# VisualPRM in full bfloat16
prm_model = AutoModel.from_pretrained(
    PRM_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"
).eval().cuda()
prm_tokenizer = AutoTokenizer.from_pretrained(PRM_PATH, trust_remote_code=True, use_fast=False)

# ─── Preprocessing Utilities ───────────────────────────────────────
def build_transform(size=INPUT_SIZE):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

from math import ceil

def dynamic_preprocess(image, image_size=INPUT_SIZE, max_num=MAX_TILES, use_thumbnail=True):
    w, h = image.size
    ar = w / h
    # build ratio candidates
    ratios = sorted({
        (i, j)
        for n in range(1, max_num+1)
        for i in range(1, n+1)
        for j in range(1, n+1)
        if 1 <= i*j <= max_num
    }, key=lambda r: r[0]*r[1])
    # pick closest aspect ratio
    best = min(ratios, key=lambda r: abs(ar - (r[0]/r[1])))
    tw, th = image_size*best[0], image_size*best[1]
    img_r = image.resize((int(tw), int(th)))
    tiles = []
    cols = int(tw // image_size)
    for idx in range(best[0]*best[1]):
        x0 = (idx % cols) * image_size
        y0 = (idx // cols) * image_size
        tiles.append(img_r.crop((x0, y0, x0+image_size, y0+image_size)))
    if use_thumbnail and len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles

# ─── Video Loading ─────────────────────────────────────────────────
def load_video(path, num_frames=NUM_FRAMES):
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    total = len(vr) - 1
    indices = np.linspace(0, total, num_frames, dtype=int)
    transform = build_transform()
    all_tiles = []
    num_patches = []
    for i in indices:
        frame = Image.fromarray(vr[i].asnumpy()).convert('RGB')
        tiles = dynamic_preprocess(frame)
        tv = torch.stack([transform(t) for t in tiles])
        num_patches.append(tv.size(0))
        all_tiles.append(tv)
    # shape: [sum(num_patches), 3, H, W]
    return torch.cat(all_tiles, dim=0), num_patches

# ─── Inference & Reranking ────────────────────────────────────────
def infer(video_path, num_candidates=3):
    # prepare video tiles
    pix, patches = load_video(video_path)
    # same bfloat16 for both models
    pix = pix.to(torch.bfloat16).contiguous().cuda()

    # build prompt
    prompt = ''.join(f"Frame{i+1}: <image>\n" for i in range(NUM_FRAMES))
    prompt += (
        "Describe what is happening in this video. Provide details on foreground, background, people, objects, "
        "and actions."
    )
    # truncate to model's max context
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][:MAX_CONTEXT]
    prompt = tokenizer.decode(ids, skip_special_tokens=True)

    # generate multiple responses
    cands = []
    for _ in range(num_candidates):
        resp, _ = model.chat(
            tokenizer,
            pix,
            prompt,
            GEN_CONFIG,
            history=None,
            return_history=True
        )
        cands.append(resp)

    # rerank with PRM
    best = prm_model.select_best_response(
        tokenizer=prm_tokenizer,
        question=prompt,
        response_list=cands,
        pixel_values=pix,
        num_patches_list=patches,
        return_scores=True
    )

    print("\nPrompt:\n", prompt)
    for i, (r, sc) in enumerate(best, 1):
        print(f"Candidate {i} (score={sc:.3f}): {r.strip()}")
    print("\nBest Response:\n", best[0][0])

if __name__ == '__main__':
    infer("TNS_0119_V.mp4")
