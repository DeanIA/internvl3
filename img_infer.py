import math
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig

# ─── Config ───────────────────────────────────────────────────────────────
MODEL_NAME   = "OpenGVLab/InternVL3-1B"
INPUT_SIZE   = 448
MAX_PATCHES  = 12
GEN_CONFIG   = dict(max_new_tokens=1024, do_sample=True)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ─── Split Model Across GPUs ──────────────────────────────────────────────
def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers

    # Distribute layers evenly, but give half of GPU0 to the vision part
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

    layer_cnt = 0
    for gpu_idx, count in enumerate(num_layers_per_gpu):
        for _ in range(count):
            device_map[f'language_model.model.layers.{layer_cnt}'] = gpu_idx
            layer_cnt += 1

    # Pin all vision & shared embeddings to GPU0
    vision_keys = [
        'vision_model',
        'mlp1',
        'language_model.model.tok_embeddings',
        'language_model.model.embed_tokens',
        'language_model.output',
        'language_model.model.norm',
        'language_model.model.rotary_emb',
        'language_model.lm_head',
        f'language_model.model.layers.{num_layers - 1}'
    ]
    for key in vision_keys:
        device_map[key] = 0

    return device_map

# ─── Image Preprocessing ─────────────────────────────────────────────────
def build_transform():
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def dynamic_preprocess(image, image_size=448, max_num=12):
    orig_w, orig_h = image.size
    aspect = orig_w / orig_h

    # find best grid (i × j) closest to aspect, with i*j ≤ max_num
    best, best_diff = (1,1), float('inf')
    for i in range(1, max_num+1):
        for j in range(1, max_num+1):
            if i*j > max_num: continue
            diff = abs(aspect - (i/j))
            if diff < best_diff:
                best, best_diff = (i,j), diff

    gw, gh = best
    new_w, new_h = image_size * gw, image_size * gh
    image = image.resize((new_w, new_h))

    tiles = []
    for y in range(gh):
        for x in range(gw):
            box = (x*image_size, y*image_size, (x+1)*image_size, (y+1)*image_size)
            tiles.append(image.crop(box))
    return tiles

def load_image(path, image_size=448, max_num=12):
    img = Image.open(path).convert('RGB')
    tiles = dynamic_preprocess(img, image_size=image_size, max_num=max_num)
    tfm = build_transform()
    return torch.stack([tfm(t) for t in tiles])

# ─── Inference ────────────────────────────────────────────────────────────
def infer(image_path):
    device_map = split_model(MODEL_NAME)
    model     = AutoModel.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    use_flash_attn=True,
                    trust_remote_code=True,
                    device_map=device_map
                ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)

    pixel_values = load_image(image_path, max_num=MAX_PATCHES)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()

    question = "<image>\nDescribe the image in detail. Mention background, people, objects, and any actions."
    response, _ = model.chat(tokenizer, pixel_values, question, GEN_CONFIG, history=None, return_history=True)

    print("User:", question)
    print("Assistant:", response)

if __name__ == "__main__":
    infer("TNS_0957_I.jpg")