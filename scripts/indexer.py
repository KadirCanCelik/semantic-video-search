import cv2
import torch
import json
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --- AYARLAR ---
VIDEO_PATH = "test_video.mp4"
OUTPUT_FILE = "video_index.json"
SAMPLE_RATE = 1

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Processor: {device.upper()} (model loading here...)")
    
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    print("✅ Model loaded.")

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if not cap.isOpened():
        print(f"❌ ERROR: '{VIDEO_PATH}' video not found. Please check the file name.")
        return

    frame_interval = int(fps / SAMPLE_RATE)
    video_data = []
    current_frame = 0

    print(f"🎬 Video is Processing... (FPS: {fps:.2f})")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            

        if current_frame % frame_interval == 0:

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            

            inputs = processor(text=None, images=pil_image, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            

            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            
            timestamp = current_frame / fps
            vector_list = image_features.cpu().numpy()[0].tolist()
            
            video_data.append({
                "timestamp": timestamp,
                "vector": vector_list
            })
            
            print(f"   ⏱️  Processed in {timestamp:.1f} seconds . (Vektor size: {len(vector_list)})")
            
        current_frame += 1
        
    cap.release()

    with open(OUTPUT_FILE, "w") as f:
        json.dump(video_data, f)
    
    print(f"\n🎉 Done! {len(video_data)} frame were processed and saved to the  '{OUTPUT_FILE}' file.")

if __name__ == "__main__":
    main()