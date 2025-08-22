##### AI2d 시스템 설계

###### 1) 사전 요건 (Prerequisites)

**하드웨어**
- GPU: RTX 2060(6GB) 이상 → RTX 4080(16GB+) 확장 가능
- RAM: 16GB 이상 권장
- 저장공간: 모델/LoRA/캐시 포함 최소 수 GB

**소프트웨어**
- OS: Windows 10/11, Linux, 또는 WSL2
- Python: 3.10 ~ 3.11 권장
- CUDA/cuDNN: PyTorch 호환 버전 (예: CUDA 11.8)
- NVIDIA 드라이버: 최신 버전

**Python 패키지**
```bash
# PyTorch(CUDA 지원)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Diffusers 스택
pip install diffusers transformers accelerate safetensors

# 성능/유틸
pip install xformers==0.0.23.post1  # 메모리 효율 개선
pip install pillow numpy opencv-python
모델/리소스

Stable Diffusion v1.5 (FP16 가중치)

ControlNet(OpenPose 모델)

IP-Adapter (스타일/아이덴티티 임베딩)

(선택) LoRA (캐릭터 전용 경량 파인튜닝)

VLM (이미지→텍스트 프롬프트 보정, keyframe마다 실행)

2) 준비 단계 (Setup)
GPU 확인: torch.cuda.is_available() / nvidia-smi

Diffusers 파이프라인 로드: torch_dtype=torch.float16, device="cuda"

성능 옵션:

pipe.enable_attention_slicing()

pipe.enable_vae_slicing()

pipe.vae.enable_tiling()

VRAM 부족 시: accelerate.enable_sequential_cpu_offload()

스케줄러 선택:

RTX 2060: DPMSolver++ 또는 Euler a, 15–22 steps 권장

참조/포즈 입력 준비:

Reference 이미지 → VLM으로 초기 프롬프트 P0 도출

Reference → IP-Adapter 임베딩 Eref 생성

Mediapipe/OpenPose → 매 프레임 스켈레톤 생성 → ControlNet 입력용 Cpose_t

해상도 정규화: 512×512 기준

3) 입력 파이프라인 (Reference & Pose)
Reference → P0 / Eref

VLM으로 전체 이미지 요약 → 초기 프롬프트 P0

IP-Adapter로 시각 임베딩 Eref

Pose → ControlNet 조건

영상/캠/키프레임 포즈 → OpenPose 맵

관절 라인: 팔/다리, 코어

해상도 매칭: 512×512

4) 실행 파이프라인 (RTX 2060 최소 동작 ≥1 fps)
초기화

해상도: 512×512

배치: 1 (OOM 없으면 2까지)

ControlNet 강도: 0.7–1.0

num_inference_steps: 15–22

guidance_scale: 5–7.5

초기 latent L0: 랜덤, 가능 시 이전 프레임 latent 재활용

실시간 루프:

Lt-1 불러오기 (첫 프레임은 L0)

ControlNet 조건 Cpose_t 생성

SD+ControlNet 추론(FP16)

입력: Lt-1 + Pn + Cpose_t (+ Eref)

출력: Lt 및 디코드 이미지 It

It 출력/표시/저장

VLM 평가 (간헐적)

매 N 프레임/keyframe만 실행

It → 프롬프트 Pn 업데이트, 속성 유지 보정

강도/스텝 동적 조정

포즈 불안정 시 강도 ↑ (최대 1.0)

안정 시 스텝 ↓

latents 업데이트

Lt → 다음 프레임 Lt-1 재사용

주기적 리셋으로 drift 방지

성능/메모리 가이드 (2060)

FP16 → attention/vae slicing → vae tiling 우선

부족 시: inference steps ↓ → 해상도 ↓ → CPU offload 순으로 조정

VLM 호출은 keyframe만 실행

5) RTX 4080 전환 (≈24 fps)
모델/가중치 동일

배치: 8–12

steps: 12–18

slicing/오프로드 대부분 해제

Flash-Attn/SDPA 가능 시 활용

ControlNet 강도: 0.7–1.0

VLM 평가 주기 단축 가능

6) 검증 & 로깅
FPS/지연: 프레임 타임 측정 (전체/모델/디코더/입출력 분리)

일관성: 얼굴/의상/헤어 색상 히스토그램/ΔE

품질: 키프레임 스냅샷, Pn 변화량 기록

안정성: OOM 이벤트, GC 시간, steps/강도 변화 기록

7) (선택) LoRA 경량 학습 파이프라인
데이터: 생성 영상/스틸 + 캡션/속성 라벨

학습 설정: 512×512, FP16, batch 1–2, 보수적 lr/scheduler

학습 시간: 2060에서는 느림 → 에폭/스텝 낮게, 주기적 검증

배포: 동일 추론 파이프라인에 LoRA merge/attach

8) 최소 체크리스트 (RTX 2060)
PyTorch(CUDA) + diffusers 설치 및 GPU 인식

SD/ControlNet/IP-Adapter 로드(FP16)

enable_attention_slicing / vae_slicing / vae tiling 적용

해상도: 512, batch=1, steps 15–22

ControlNet 강도: 0.7–1.0

latent 재활용 루틴 작동 확인

VLM 평가는 keyframe만 호출

OOM 발생 시:

steps ↓

해상도 ↓

CPU offload