# sd_controlnet.py
import os
import time
import torch
import numpy as np
import cv2
from PIL import Image
from contextlib import nullcontext
from typing import Optional, List

from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
)
from controlnet_aux import OpenposeDetector

# ----------------------------
# 유틸
# ----------------------------
def log(s: str):
    print(f"[INFO] {s}")

def set_seed(seed: Optional[int] = None) -> torch.Generator:
    g = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    if seed is None:
        seed = torch.seed() % (2**32)
    g.manual_seed(int(seed))
    return g

def to_pil(img_bgr: np.ndarray, size=(512, 512)) -> Image.Image:
    img = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def read_bgr(path: str, size=(512, 512)) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

# ----------------------------
# 모델 로드
# ----------------------------
def load_pipelines(model_id="runwayml/stable-diffusion-v1-5",
                   controlnet_id="lllyasviel/sd-controlnet-openpose",
                   use_half=True, scheduler="dpmpp",
                   slicing=True, tiling=True, cpu_offload=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if use_half and device=="cuda" else torch.float32
    log(f"Device={device}, dtype={'fp16' if dtype==torch.float16 else 'fp32'}")

    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)

    pipe_t2i = StableDiffusionControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=dtype,
        safety_checker=None, feature_extractor=None
    )
    if scheduler=="dpmpp":
        pipe_t2i.scheduler = DPMSolverMultistepScheduler.from_config(pipe_t2i.scheduler.config)
    else:
        pipe_t2i.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_t2i.scheduler.config)

    try: pipe_t2i.enable_xformers_memory_efficient_attention(); log("xformers enabled")
    except: pass
    if slicing:
        pipe_t2i.enable_attention_slicing()
        pipe_t2i.enable_vae_slicing()
    if tiling:
        pipe_t2i.vae.enable_tiling()
    if cpu_offload:
        pipe_t2i.enable_sequential_cpu_offload()
    pipe_t2i = pipe_t2i.to(device)

    pipe_i2i = StableDiffusionControlNetImg2ImgPipeline(
        vae=pipe_t2i.vae, text_encoder=pipe_t2i.text_encoder, tokenizer=pipe_t2i.tokenizer,
        unet=pipe_t2i.unet, controlnet=pipe_t2i.controlnet,
        scheduler=pipe_t2i.scheduler, safety_checker=None, feature_extractor=None
    )
    if slicing:
        pipe_i2i.enable_attention_slicing()
        pipe_i2i.enable_vae_slicing()
    if tiling:
        pipe_i2i.vae.enable_tiling()
    try: pipe_i2i.enable_xformers_memory_efficient_attention()
    except: pass
    if cpu_offload:
        pipe_i2i.enable_sequential_cpu_offload()
    pipe_i2i = pipe_i2i.to(device)

    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    return pipe_t2i, pipe_i2i, openpose

# ----------------------------
# OpenPose 제어 이미지
# ----------------------------
def get_pose_control_image(openpose: OpenposeDetector, frame_bgr: np.ndarray) -> Image.Image:
    frame_pil = to_pil(frame_bgr)
    pose_pil = openpose(frame_pil)
    return pose_pil.convert("RGB") if pose_pil.mode!="RGB" else pose_pil
