import torch
import clip
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# TO ADD:
# Gradient Checkpointing
# Filter out bias from weight decay
# Decaying learning rate with cosine schedule
# Half-precision Adam statistics
# Half-precision stochastically rounded text encoder weights were used

# Set batch size
BATCH_SIZE = 8
EPOCH = 3 # Replace 10 with the desired number of epochs

# Load CLIP model
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)

# Read data
data = pd.read_csv('data_new.csv')

class ImageTitleDataset(Dataset):
    def __init__(self, list_image_path, list_txt):
        self.image_path = list_image_path
        self.title = clip.tokenize(list_txt)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path[idx]))
        title = self.title[idx]
        return image, title

# Create dataset and dataloader
list_image_path = data['filepath'].tolist()
list_txt = data['caption'].tolist()
dataset = ImageTitleDataset(list_image_path, list_txt)
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Gradient checkpointing
def clip_forward(model, images, texts):
    return model(images, texts)

# Filter out bias from weight decay
params = [
    {"params": [p for name, p in model.named_parameters() if 'bias' not in name and 'visual' in name]},
    {"params": [p for name, p in model.named_parameters() if 'bias' in name and 'visual' in name], "weight_decay": 0.0}
]

optimizer = optim.Adam(params, lr=5e-6, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)

# Set requires_grad=True for all model parameters
for param in model.parameters():
    param.requires_grad = True

# Training loop
for epoch in range(EPOCH):
    model.train()  # Set the model to train mode

    total_loss = 0.0

    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        images, texts = batch
        images = images.to(device)
        texts = texts.to(device)

        # Forward pass
        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print the training progress
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch [{epoch+1}/{EPOCH}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {avg_loss:.8f}")

    # Training statistics for the epoch
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{EPOCH}], Average Loss: {avg_loss:.8f}")
    
    # Save the model after each epoch
    torch.save(model.state_dict(), f"trained_model_epoch_{epoch+1}.pth")