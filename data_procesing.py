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

spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self,freq_threshold):
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        self.stoi = {"<PAD>":0,"<SOS>":1,"<EOS>":2,"<UNK>":3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocabulary(self,sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self,text):
        tokenized_text = self.tokenizer(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class FlickrDataset(Dataset):
    def __init__(self,root_dir,captions_file,transform=None,freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        caption = self.captions[idx]
        img_id = self.imgs[idx]
        img = Image.open(os.path.join(self.root_dir,img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img,torch.tensor(numericalized_caption)

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])

def show_image(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

class CapsCollate:
    """
    Custom collate function for DataLoader to handle images and padded captions.
    
    Args:
        pad_idx (int): The padding index for captions.
        batch_first (bool): If True, captions will have shape (batch_size, seq_len).
                            If False, captions will have shape (seq_len, batch_size).
    """
    def __init__(self, pad_idx: int, batch_first: bool = False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
    
    def __call__(self, batch):
        """
        Collates a batch of images and captions, applying padding to captions.
        """
        if not batch or not isinstance(batch, list):
            raise ValueError("Input batch must be a non-empty list of (image, caption) tuples.")
        
        # Process images
        imgs = torch.stack([item[0] for item in batch], dim=0)  # Shape: (batch_size, C, H, W)

        # Process captions
        captions = [item[1] for item in batch]
        targets = pad_sequence(captions, batch_first=self.batch_first, padding_value=self.pad_idx)
        
        return imgs, targets

def get_data_loader(root_folder,caption_file,transform,batch_size=256,num_workers=4):
    dataset = FlickrDataset(
        root_dir = root_folder,
        captions_file = caption_file,
        transform = transform
    )

    pad_idx = dataset.vocab.stoi["<PAD>"]
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
    )

    return data_loader,dataset

'''

data_location ='data/'

caption_file = data_location + 'captions.txt'
df = pd.read_csv(caption_file)
# print("There are {} image to captions".format(len(df)))
# print(df.head(7))


# text ="A dog is playing with a ball"
# print([token.text.lower() for token in spacy_eng.tokenizer(text)])
'''
