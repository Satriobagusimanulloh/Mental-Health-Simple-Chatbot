import numpy as np
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNetwork
from text_preprocessing import tokenize, stem, bag_of_words
from customdataset import CustomDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("../data/intents.json", "r") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Mendapatkan tag pertanyaan. Melakukan tokenisasi pada pertanyaan dan mendapatkan semua kata pada pertanyaan
for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for question in intent["patterns"]:
        tokenized_words = tokenize(question)
        all_words.extend(tokenized_words)
        xy.append((tokenized_words, tag))

# Filter semua kata pada pertanyaan dan mengubahnya ke kata dasar. Mengurutkan himpunan unik dari all_words dan tags
ignore_words = [".", ",", "?", "!"]
all_words = [stem(word) for word in all_words if word not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (tokenized_words, tag) in xy:
    bag = bag_of_words(tokenized_words, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

input_size = len(X_train[0])
output_size = len(tags)
hidden_size = 256
batch_size = 32
learning_rate = 0.001
num_epochs = 1000

dataset = CustomDataset(X_train, y_train)

train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

model = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, num_classes=output_size)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        y_pred = model(words)
        loss = loss_fn(y_pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print(f'Final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "model_mental_health.pth"
torch.save(data, FILE)

print(f'Training complete. File saved to {FILE}')