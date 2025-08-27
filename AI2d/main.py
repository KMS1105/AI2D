import csv
import json
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from diffusers import StableDiffusionImg2ImgPipeline
import torch
import os

# --- CSV에서 키워드 로드 ---
def load_keywords_from_csv(csv_path):
    keywords = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = row['Keyword']
            words = [w.strip() for w in sentence.replace(',', ' ').split() if w.strip()]
            keywords.extend(words)
    return keywords

# --- 이미지 사이즈 맞춤 ---
def resize_image(img, target_size=(512, 512)):
    return img.resize(target_size, Image.LANCZOS)

# --- SSIM 계산 ---
def ssim_score(img1, img2):
    img1 = np.array(img1.convert("L"))
    img2 = np.array(img2.convert("L"))
    return ssim(img1, img2)

# --- 최적값 JSON 저장/불러오기 ---
def save_best_params(json_path, params):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

def load_best_params(json_path):
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# --- img2img 최적화 및 적용 ---
def optimize_img2img(init_image_path, csv_path, output_path,
                     best_params_path="./best_params.json",
                     steps=10, strengths=[0.25,0.3,0.35], guidance_scales=[7.0,7.5,8.0]):

    keywords = load_keywords_from_csv(csv_path)
    prompt = ", ".join(keywords)
    
    original = Image.open(init_image_path).convert("RGB")
    original = resize_image(original)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to(device)

    # 기존 최적값 불러오기
    best_params = load_best_params(best_params_path)
    if best_params:
        print("저장된 최적값 불러오기")
        strength = best_params["strength"]
        guidance = best_params["guidance_scale"]
        result = pipe(prompt=prompt, image=original, strength=strength, guidance_scale=guidance).images[0]
        result_resized = resize_image(result)
        result_resized.save(output_path)
        print(f"최적값 적용 완료: {output_path}, strength={strength}, guidance_scale={guidance}")
        return

    # 없으면 탐색
    best_score = -1
    best_img = None
    best_strength = None
    best_guidance = None

    for step in range(steps):
        for strength in strengths:
            for guidance in guidance_scales:
                result = pipe(prompt=prompt, image=original, strength=strength, guidance_scale=guidance).images[0]
                result_resized = resize_image(result)
                score = ssim_score(original, result_resized)
                print(f"Step {step+1}, strength={strength}, guidance={guidance}: SSIM={score:.4f}")
                if score > best_score:
                    best_score = score
                    best_img = result_resized
                    best_strength = strength
                    best_guidance = guidance

    # 결과 저장
    best_img.save(output_path)
    print(f"최종 저장: {output_path}, 최고 SSIM={best_score:.4f}")

    # 최적값 저장
    save_best_params(best_params_path, {"strength": best_strength, "guidance_scale": best_guidance})
    print(f"최적값 저장 완료: {best_params_path}")

csv_file = "./dataset/character_keywords_weighted4.csv"
init_image = "./input/damui.png"
output_image = "./result/best_result.png"
optimize_img2img(init_image, csv_file, output_image)

