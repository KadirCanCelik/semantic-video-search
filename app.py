import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import tempfile
import os


MODEL_ID = "openai/clip-vit-base-patch32"
SAMPLE_RATE = 1

st.set_page_config(page_title="AI Video Search", page_icon="🔍", layout="wide")

@st.cache_resource
def load_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"⚙️ Model Loading ... (Device: {device.upper()})")
    
    model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    
    return model, processor, device

model, processor, device = load_model()


def process_video(video_path):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / SAMPLE_RATE)
    
    video_data = []
    current_frame = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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
            video_data.append({
                "timestamp": timestamp,
                "vector": image_features.cpu().numpy()[0]
            })
            
            status_text.text(f"In Progress: {timestamp:.1f}. second")
            progress = min(current_frame / total_frames, 1.0)
            progress_bar.progress(progress)
            
        current_frame += 1
        
    cap.release()
    progress_bar.progress(100)
    status_text.text("✅ Indexing Completed!")
    return video_data

# --- (FRONTEND) ---
st.title("🎬 Semantic Video Search Engine")
st.markdown("Search for **'word'** in the video with artificial intelligence.")

uploaded_file = st.file_uploader("Load a video file (.mp4)", type=["mp4"])

if uploaded_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    if st.button("🚀 Analysis the video"):
        with st.spinner('AI is watching the video'):
            st.session_state['video_index'] = process_video(video_path)
            st.success(f"Analysis done! {len(st.session_state['video_index'])} frames were stored in the memory")

    if 'video_index' in st.session_state:
        st.divider()
        st.subheader("🔍 Search")
        
        query = st.text_input("What do you want to search?", placeholder="Example: red car, delicious pizza...")
        
        if query:

            inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                text_features = model.get_text_features(**inputs)
            text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
            text_vector = text_features.cpu().numpy()[0]

            results = []
            for frame in st.session_state['video_index']:
                frame_vector = frame["vector"]
                score = np.dot(text_vector, frame_vector)
                if score > 0.20:
                    results.append((score, frame["timestamp"]))
            
            results.sort(key=lambda x: x[0], reverse=True)


            if results:
                st.success(f"Best scenes for '{query}':")
                
                cols = st.columns(3)
                for i, (score, timestamp) in enumerate(results[:3]):
                    with cols[i]:
                        st.metric(label=f"Result #{i+1}", value=f"{timestamp:.1f} sec", delta=f"%{score*100:.1f} Similarity")

                        st.video(video_path, start_time=int(timestamp))
            else:
                st.warning("No matching scene found.")