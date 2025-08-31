import os
import argparse
import pandas as pd
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline, ControlNetModel
import cv2

# -----------------------------
# CSV에서 키워드 로드
# -----------------------------
def load_keywords_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    if 'keyword' not in df.columns:
        raise KeyError("[ERROR] CSV 파일에 'keyword' 컬럼이 없습니다.")
    return [str(row['keyword']) for _, row in df.iterrows()]

# -----------------------------
# ControlNet 포함 SD Img2Img Pipeline
# -----------------------------
def create_sd_controlnet_pipeline(model_name="runwayml/stable-diffusion-v1-5", controlnet_name=None, device="cuda"):
    if controlnet_name:
        controlnet = ControlNetModel.from_pretrained(controlnet_name)
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, controlnet=controlnet, torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    return pipe

# -----------------------------
# 시퀀스 렌더링
# -----------------------------
def render_sequence(pipe, init_image_path, npy_path=None, csv_path=None, out_dir="./result",
                    negative_prompt="", start=0, end=None, step=1,
                    seed=42, strength=0.7, guidance_scale=7.5,
                    num_inference_steps=20, controlnet_weight=1.0, resize_size=(512,512)):

    os.makedirs(out_dir, exist_ok=True)
    frames = []

    # 키워드 로드
    prompts = load_keywords_from_csv(csv_path) if csv_path else [""]

    # 초기 이미지
    init_image = Image.open(init_image_path).convert("RGB")
    if resize_size:
        init_image = init_image.resize(resize_size)

    if end is None:
        end = len(prompts)

    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    for idx in range(start, end, step):
        prompt = prompts[idx % len(prompts)]  # 안전하게 반복
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            controlnet_conditioning_scale=controlnet_weight
        ).images[0]

        frame_path = os.path.join(out_dir, f"frame_{idx:04d}.png")
        output.save(frame_path)
        frames.append(frame_path)

        print(f"[LOG] Saved frame {idx} -> {frame_path}")

    return frames

# -----------------------------
# 프레임 -> mp4 변환
# -----------------------------
def frames_to_video(frame_paths, output_path, fps=24):
    if not frame_paths:
        print("[WARN] No frames to convert to video.")
        return
    first_frame = cv2.imread(frame_paths[0])
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for f in frame_paths:
        img = cv2.imread(f)
        video.write(img)

    video.release()
    print(f"[LOG] Video saved to {output_path}")

# -----------------------------
# 메인
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=str, required=True, help="초기 이미지 경로")
    parser.add_argument("--npy", type=str, default=None, help="표정 npy 파일 경로")
    parser.add_argument("--csv", type=str, default="./dataset/character_keywords_weighted4.csv", help="키워드 CSV 경로")
    parser.add_argument("--out", type=str, default="./result", help="출력 디렉토리")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--controlnet_weight", type=float, default=1.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet", type=str, default=None)
    args = parser.parse_args()

    pipe_i2i = create_sd_controlnet_pipeline(model_name=args.model, controlnet_name=args.controlnet)

    frames = render_sequence(
        pipe_i2i,
        init_image_path=args.init,
        npy_path=args.npy,
        csv_path=args.csv,
        out_dir=args.out,
        negative_prompt="low quality, deformed, extra limbs, extra faces",
        start=args.start,
        end=args.end,
        step=1,
        seed=args.seed,
        strength=args.strength,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        controlnet_weight=args.controlnet_weight,
        resize_size=(512,512)
    )

    video_path = os.path.join(args.out, "result.mp4")
    frames_to_video(frames, video_path, fps=args.fps)
