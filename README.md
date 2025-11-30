# semantic_video_search

**Search inside videos using natural language.** This project is an AI-powered search engine that allows users to find specific scenes in a video by describing them in text (e.g., *"a red car driving,"* *"delicious pizza,"* *"dog running on beach"*). Unlike traditional keyword search, it uses **Multimodal Vector Embeddings** to understand the semantic meaning of the visual content.

## 🚀 Key Features

* **🧠 Multimodal AI:** Uses **OpenAI's CLIP (Contrastive Language-Image Pre-training)** model to map images and text into the same vector space.
* **🔍 Semantic Search:** Performs "Zero-Shot" searching. No manual tagging or training required.
* **⚡ GPU Acceleration:** Optimized for NVIDIA RTX GPUs using CUDA for fast inference and indexing.
* **🎞️ Automated Indexing:** Automatically extracts frames from video and converts them into vector embeddings.
* **🖥️ Modern UI:** Interactive web interface built with **Streamlit**.

## 🛠️ System Architecture

The project consists of two main pipelines: **Indexing** and **Searching**.

1.  **Indexing Pipeline:** The video is processed frame-by-frame. Each frame is passed through the CLIP Image Encoder to generate a 512-dimensional vector.
2.  **Search Pipeline:** The user's text query is passed through the CLIP Text Encoder. The system calculates the **Cosine Similarity** between the text vector and frame vectors to rank the most relevant scenes.


## 🧪 Testing with CLI (Optional)
If you prefer using the terminal instead of the web UI, you can use the scripts in the scripts/ folder:

* **python scripts/indexer.py** "This generates a video_index.json file."
* **python scripts/search_engine.py**

## 📸 Demo

![Project Demo](images/demo1.png)
![Project Demo](images/demo2.png)
![Project Demo](images/demo3.png)
