from flask import Flask, request, render_template, redirect, url_for
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T
import pandas as pd
import os
from collections import Counter
import spacy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt 
from model import EncoderDecoder , DecoderRNN, EncoderCNN  # Import mô hình của bạn
from data_procesing import get_data_loader
import pickle

data_location = 'data'
BATCH_SIZE = 128
NUM_WORKER = 4
vocab_file = 'vocab.pkl'
with open(vocab_file, 'rb') as f:
    vocab = pickle.load(f)
    
print(f"Vocabulary loaded with size: {len(vocab)}")

# Tiền xử lý ảnh
transforms = T.Compose([
    T.Resize(226),                     
    T.RandomCrop(224),                 
    T.ToTensor(),                               
    T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])

# Khởi tạo Flask app
app = Flask(__name__)

# Thiết bị (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load mô hình đã train
model_path = r"checkpoint\attention_model_state.pth"
model_state = torch.load(model_path, map_location=device)

# Khởi tạo mô hình
model = EncoderDecoder(
    embed_size=model_state['embed_size'],
    vocab_size=model_state['vocab_size'],
    attention_dim=model_state['attention_dim'],
    encoder_dim=model_state['encoder_dim'],
    decoder_dim=model_state['decoder_dim']
)
model.load_state_dict(model_state['state_dict'])
model = model.to(device)
model.eval()

transforms = T.Compose([
    T.Resize(226),                     
    T.RandomCrop(224),                 
    T.ToTensor(),                               
    T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])
# Hàm tiền xử lý ảnh
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transforms(image).unsqueeze(0)  # Thêm batch dimension
    return image

# Hàm inference
def generate_caption(image_path, vocab):
    image = process_image(image_path)
    image = image.to(device)
    with torch.no_grad():
        features = model.encoder(image)
        caption = model.decoder.generate_caption(features, vocab= vocab)  # Implement hàm này
    return caption

# Trang chủ
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            # Lưu file ảnh
            image_path = "static/" + file.filename
            file.save(image_path)

            # Generate caption
            caption = generate_caption(image_path, vocab = vocab)  # Thay vocab nếu cần
            caption = caption[0][0:-1]
            caption = ' '.join(caption)
            return render_template("index.html", image_path=image_path, caption=caption)

    return render_template("index.html", image_path=None, caption=None)

if __name__ == "__main__":
    app.run(debug=True)
