import torch
import json
import numpy as np
from transformers import CLIPProcessor, CLIPModel

# --- AYARLAR ---
INDEX_FILE = "video_index.json"
MODEL_ID = "openai/clip-vit-base-patch32"

def load_data():

    try:
        with open(INDEX_FILE, "r") as f:
            data = json.load(f)
        print(f"📂 A database of {len(data)} frames has been loaded")
        return data
    except FileNotFoundError:
        print(f"❌ ERROR: '{INDEX_FILE}' not found. indexer.py must be run first!")
        return []

def seconds_to_time(seconds):

    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Search engine is running... (Device: {device.upper()})")
    
    model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    video_data = load_data()
    if not video_data:
        return

    print("\n🔎 Enter the text to search (Press 'q' to quit)")
    print("-" * 50)

    while True:
        query_text = input("\Query: ")
        if query_text.lower() == 'q':
            print("Quit...")
            break
            
        inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        text_vector = text_features.cpu().numpy()[0] 

        results = []
        for frame in video_data:
            frame_vector = np.array(frame["vector"])
            
            score = np.dot(text_vector, frame_vector)
            
            if score > 0.21: 
                results.append((score, frame["timestamp"]))
        
        results.sort(key=lambda x: x[0], reverse=True)

        if results:
            print(f"\n Best results for ✅ '{query_text}':")
            for score, timestamp in results[:3]:
                time_str = seconds_to_time(timestamp)
                print(f"   👉 Time : {time_str} (Second: {timestamp:.1f}) | Similarity: %{score*100:.1f}")
        else:
            print("❌ No matches found. Try a more general expression.")

if __name__ == "__main__":
    main()