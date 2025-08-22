# main.py
import os
import time
from vlm_prompt import generate_prompt_from_image
from sd_controlnet import (
    load_pipelines, read_bgr, get_pose_control_image, log, set_seed
)

def generate_loop(pipe_t2i, pipe_i2i, openpose,
                  image_paths, frames, strength, steps, guidance, seed,
                  out_dir, use_feedback=False, feedback_denoise=0.25, batch=1,
                  neg_prompt="blurry, lowres, deformed, bad anatomy, low quality"):

    os.makedirs(out_dir, exist_ok=True)
    gen = set_seed(seed)
    total = len(image_paths) if frames<=0 else min(frames, len(image_paths))
    prev_img = None
    times = []

    for idx, path in enumerate(image_paths[:total]):
        t0 = time.time()
        frame_bgr = read_bgr(path)
        control_pil = get_pose_control_image(openpose, frame_bgr)

        # ✅ 여기서 VLM 불러오기
        prompt = generate_prompt_from_image(control_pil)
        print(f"[DEBUG] Frame {idx} Prompt: {prompt}")

        result = pipe_t2i(
            prompt=[prompt]*batch,
            negative_prompt=[neg_prompt]*batch if neg_prompt else None,
            image=[control_pil]*batch,
            controlnet_conditioning_scale=strength,
            generator=gen,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=512,
            width=512
        )

        for b, img in enumerate(result.images):
            img.save(os.path.join(out_dir, f"frame_{idx:05d}_{b}.png"))

        dt = time.time()-t0
        times.append(dt)
        log(f"[{idx+1}/{total}] {os.path.basename(path)} | {dt:.2f}s | FPS={1/dt:.2f}")

    if times:
        avg = sum(times)/len(times)
        log(f"Average FPS: {1/avg:.2f}")

def main():
    input_path = "./input/damui.png"

    pipe_t2i, pipe_i2i, openpose = load_pipelines()

    generate_loop(
        pipe_t2i, pipe_i2i, openpose,
        image_paths=[input_path],
        frames=1,
        strength=0.85,
        steps=18,
        guidance=5.5,
        seed=42,
        out_dir="./out",
        use_feedback=False
    )

if __name__=="__main__":
    main()
