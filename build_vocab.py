import pickle
from data_procesing import FlickrDataset
import pandas as pd 
import os
from collections import Counter
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt 
# Tạo tập dữ liệu FlickrDataset
root_folder = 'data/images'  # Cập nhật đường dẫn đúng
caption_file = 'data/captions.txt'  # Cập nhật đường dẫn đúng
transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

# Khởi tạo dataset
dataset = FlickrDataset(
    root_dir=root_folder,
    captions_file=caption_file,
    transform=transform
)

# Lưu vocab vào file pickle
vocab_file = 'vocab.pkl'  # Tên file để lưu vocab
with open(vocab_file, 'wb') as f:
    pickle.dump(dataset.vocab, f)

print(f"Vocabulary has been saved to {vocab_file}.")
