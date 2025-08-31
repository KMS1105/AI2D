import os
import argparse
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from transformers import logging as tf_logging
import logging

# transformers 토큰 로깅 끄기
tf_logging.set_verbosity_error()
logging.getLogger("diffusers").setLevel(logging.ERROR)

# ----------------------------
# 유틸리티 함수
# ----------------------------
def log(msg):
    print(f"[LOG] {msg}")

def load_keywords_from_csv(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    # weight 적용해서 키워드 합치기
    prompt_list = []
    for _, row in df.iterrows():
        word = row['Keyword']
        weight = row.get('weight', 1.0)
        if weight > 0:
            prompt_list.append(f"{word}:{weight}")
    return ", ".join(prompt_list)

def facemesh_to_control_image(landmarks, size=(512,512), normalized=True):
    from PIL import ImageDraw
    img = Image.new("RGB", size, (0,0,0))
    draw = ImageDraw.Draw(img)
    w, h = size
    for lm in landmarks:
        x, y = lm[0], lm[1]
        if normalized:
            x, y = int(x*w), int(y*h)
        draw.ellipse((x-1, y-1, x+1, y+1), fill=(255,255,255))
    return img

def run_img2img_frame(pipe, init_img, control_img, prompt, negative_prompt=None,
                      seed=1234, strength=0.25, guidance_scale=8.0,
                      num_inference_steps=20, controlnet_weight=1.0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_img,
        control_image=control_img,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        controlnet_conditioning_scale=controlnet_weight
    )
    return result.images[0]

# ----------------------------
# 시퀀스 렌더 함수
# ----------------------------
def render_sequence(
    pipe_i2i,
    init_image_path: str,
    npy_path: str,
    csv_path: str,
    out_dir: str,
    negative_prompt: str = None,
    start: int = 0,
    end: int = None,
    step: int = 1,
    seed: int = 1234,
    strength: float = 0.5,
    guidance_scale: float = 8.0,
    num_inference_steps: int = 20,
    controlnet_weight: float = 2.0,
    resize_size: tuple = (512,512),
):
    os.makedirs(out_dir, exist_ok=True)
    prompt = load_keywords_from_csv(csv_path)
    log(f"Prompt: {prompt[:200]}...")

    init_pil = Image.open(init_image_path).convert("RGB").resize(resize_size)
    npy = np.load(npy_path)
    total = len(npy)
    if end is None:
        end = total

    log(f"Loaded npy frames: {total}. Rendering frames {start}..{end} step {step}")

    frame_paths = []
    total_frames = min(end - start, total - start)
    for count, idx in enumerate(range(start, min(end, total), step), 1):
        lm = npy[idx]
        control_img = facemesh_to_control_image(lm, size=resize_size, normalized=True)
        out_img = run_img2img_frame(
            pipe_i2i,
            init_pil,
            control_img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed + idx,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            controlnet_weight=controlnet_weight,
        )
        out_path = os.path.join(out_dir, f"frame_{idx:05d}.png")
        out_img.save(out_path)
        frame_paths.append(out_path)

        overall_progress = count / total_frames * 100
        print(f"\r전체 진행: {overall_progress:.1f}% [{count}/{total_frames}]", end="")

    print()
    log(f"Rendering finished. {len(frame_paths)} frames saved to {out_dir}")
    return frame_paths

# ----------------------------
# 메인 실행부
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=str, default="./input/damui.png", help="초기 이미지 경로")
    parser.add_argument("--npy", type=str, default="./face_mesh/face_mesh_1min.npy", help="표정 npy 파일 경로")
    parser.add_argument("--csv", type=str, default="./dataset/character_keywords_weighted4.csv", help="CSV 키워드 파일")
    parser.add_argument("--out", type=str, default="./result", help="결과 저장 폴더")
    parser.add_argument("--seed", type=int, default=1234, help="시드값")
    parser.add_argument("--strength", type=float, default=0.25, help="img2img strength")
    parser.add_argument("--guidance", type=float, default=8.0, help="guidance scale")
    parser.add_argument("--steps", type=int, default=20, help="num inference steps")
    parser.add_argument("--step", type=int, default=1, help="프레임 step")
    parser.add_argument("--controlnet_weight", type=float, default=1.0, help="ControlNet weight")
    parser.add_argument("--start", type=int, default=0, help="렌더 시작 프레임")
    parser.add_argument("--end", type=int, default=None, help="렌더 종료 프레임")
    args = parser.parse_args()

    log("Loading Stable Diffusion img2img + ControlNet pipeline...")
    pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    frames = render_sequence(
        pipe_i2i,
        init_image_path=args.init,
        npy_path=args.npy,
        csv_path=args.csv,
        out_dir=args.out,
        negative_prompt="low quality, deformed, extra limbs, extra faces",
        start=300,
        end=600,
        step=args.step,
        seed=args.seed,
        strength=args.strength,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        controlnet_weight=args.controlnet_weight,
        resize_size=(512,512),
    )
