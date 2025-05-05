# 🧠 Mini Multimodal Transformer Project: Image Captioning + Visual Question Answering

This project demonstrates how to combine vision and language models to build a **multimodal transformer** system that:
- Generates image captions
- Answers natural language questions about an image

Runs locally on macOS with CPU support using Hugging Face Transformers.

---

## 📦 Features
- ✅ Image encoder using CLIP or BLIP
- ✅ Text decoder using T5 or BLIP's built-in decoder
- ✅ Visual Question Answering (VQA)
- ✅ Extensible for your own photos or webcam input
- ✅ Minimal dependencies

---

## 🧠 Concepts Covered
- Vision-Language Transformers
- Cross-modal feature fusion
- Model freezing and modularity
- Image-text prompting
- Hugging Face pipelines for multimodal tasks

---

## 🖥️ Requirements

- Python 3.8+
- macOS with 8GB+ RAM
- Pip packages:

```bash
pip install torch torchvision transformers datasets pillow matplotlib
````

---

## 🚀 Quick Start (BLIP: Pretrained Captioning + VQA)

### 📸 Captioning an Image

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image = Image.open("your_image.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs)

print(processor.decode(out[0], skip_special_tokens=True))
```

---

### ❓ Visual Question Answering

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base")

image = Image.open("your_image.jpg").convert("RGB")
question = "What is the person doing?"

inputs = vqa_processor(images=image, text=question, return_tensors="pt")
out = vqa_model.generate(**inputs)

print(vqa_processor.decode(out[0], skip_special_tokens=True))
```

---

## 🧪 Advanced: Build Your Own Model from CLIP + T5

To go deeper, try:

* Extracting features from `CLIPModel`
* Projecting image embeddings into T5 encoder space
* Using a simple MLP or attention layer as a bridge

Example (vision encoder only):

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("your_image.jpg").convert("RGB")
inputs = clip_processor(images=image, return_tensors="pt")

with torch.no_grad():
    image_features = clip_model.get_image_features(**inputs)
```

---

## 🌍 Extensions

* 🎤 Add Whisper to transcribe spoken questions
* 🖼️ Build a dataset from personal images + captions
* 🌐 Use Streamlit or Gradio to build a web app
* 🌍 Add multilingual captioning with mT5

---

## 📁 Project Structure

```
multimodal-transformer/
├── README.md
├── image_captioning.py
├── vqa.py
├── advanced_clip_t5.py
└── your_image.jpg
```

---

## 📚 References

* [Hugging Face Transformers](https://huggingface.co/docs/transformers)
* [BLIP paper](https://arxiv.org/abs/2201.12086)
* [CLIP paper](https://arxiv.org/abs/2103.00020)

```

---
