import numpy as np
import pytest
import torch
from PIL import Image
import csv
import clip
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load your ViT-B-32 checkpoint
model_path = "trained_model_epoch_3.pth"  # Replace with the actual path to your checkpoint file
checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint

# Load the model
model, preprocess = clip.load("ViT-B/16", device=device)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# Load the CSV file
csv_file = "testset.csv"

total_images = 0
correct_predictions = 0

instructions = ["remove snow from the image", "remove raindrops from the image", "remove haze from the image", "remove rain from the image"]

with open(csv_file, "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        image_path = row[0]
        instruction = row[1]

        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text = clip.tokenize(instructions).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            predicted_label = torch.argmax(logits_per_image)
            # Print the label probabilities
            print("Label probs:", probs)

        total_images += 1
        predicted_instruction = instructions[predicted_label]
        
        print(predicted_instruction, "  ", instruction)
        if predicted_instruction == instruction:
            correct_predictions += 1
            print(f"Image {image_path}: Correct prediction")
        else:
            print(f"Image {image_path}: Wrong prediction")

accuracy = correct_predictions / total_images
print(f"Accuracy: {accuracy * 100:.2f}%")