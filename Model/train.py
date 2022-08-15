import json
from nltk_util import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

f = open('bot_data_1_0.json',encoding="utf8")
intents_data = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents_data['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        word = tokenize(pattern)
        all_words.extend(word)
        xy.append((word, tag))
        
ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label)
    
X_train = np.array(X_train)
y_train = np.array(y_train)

#HyperParameter    
batch_size = 8
hidden_size = 20

output_size = len(tags)
print("output_size: ",output_size)

input_size = len(X_train[0])
print("input_size", input_size)

learning_rate = 0.001
num_epochs = 1000 

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples


    
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, 
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

model = NeuralNet(input_size, hidden_size, output_size)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        labels = labels.to(dtype=torch.long)

        #forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        #backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if  (epoch +1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
print(f'training complete. file saved to {FILE}')
