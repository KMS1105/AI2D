import ollama
import cv2
import re
import csv
import os
from collections import Counter, defaultdict

def get_keywords(img_segment, num_keywords, model="llava-phi3:3.8b-mini-fp16"):
    # 임시 파일로 저장 (각 segment마다 저장 후 전달)
    temp_path = "./temp_segment.png"
    cv2.imwrite(temp_path, img_segment)

    # Ollama에 이미지와 함께 전달
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a vision-language assistant. "
                    "The user provides an anime/manga-style character illustration. "
                    "Your job is ONLY to output concise visual keywords about appearance. "
                    "Do not include conversational phrases. "
                    "List only keywords separated by commas."
                ),
            },
            {
                "role": "user",
                "content": f"Extract {num_keywords} descriptive keywords from this image.",
                "images": [temp_path]
            }
        ]
    )

    text = response["message"]["content"]
    keywords = re.split(r"[,;\n]", text)
    return [kw.strip().lower() for kw in keywords if kw.strip()][:num_keywords]

def extract_keywords_weighted(image_path, model="llava-phi3:3.8b-mini-fp16",
                              whole_keywords=50, quadrant_keywords=20,
                              repeats=3, csv_path="keywords_weighted.csv"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w, _ = img.shape
    quadrants = {
        "TopLeft": img[0:h//2, 0:w//2],
        "TopRight": img[0:h//2, w//2:w],
        "BottomLeft": img[h//2:h, 0:w//2],
        "BottomRight": img[h//2:h, w//2:w]
    }

    counters = {"WholeImage": Counter(),
                "TopLeft": Counter(),
                "TopRight": Counter(),
                "BottomLeft": Counter(),
                "BottomRight": Counter()}

    for i in range(repeats):
        print(f"Iteration {i+1}/{repeats} - Whole Image")
        whole_kw = get_keywords(img, whole_keywords, model)
        counters["WholeImage"].update(whole_kw)

        for name, quad_img in quadrants.items():
            print(f"Iteration {i+1}/{repeats} - {name}")
            kw = get_keywords(quad_img, quadrant_keywords, model)
            counters[name].update(kw)

    # 통합 가중치 계산
    total_counter = Counter()
    region_detail = defaultdict(lambda: defaultdict(int))
    for region, counter in counters.items():
        for kw, count in counter.items():
            total_counter[kw] += count
            region_detail[kw][region] = count

    # CSV 저장
    folder = os.path.dirname(csv_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Keyword", "TotalWeight", "RegionCounts"])
        for kw, count in total_counter.items():
            total_weight = count / (repeats * 5)  # 반복 × 5개 영역
            region_counts = "; ".join(f"{r}:{c}" for r, c in region_detail[kw].items())
            writer.writerow([kw, total_weight, region_counts])

    print(f"CSV saved to {csv_path}")
    return total_counter, region_detail

if __name__ == "__main__":
    image_path = "./input/damui.png"
    csv_file = "character_keywords_weighted.csv"
    total_counter, region_detail = extract_keywords_weighted(image_path, csv_path=csv_file)
