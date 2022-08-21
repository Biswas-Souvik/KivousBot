import torch.nn as nn
import json, torch, random
from nltk_util import tokenize, stem, tf_idf

f = open('app/bot_data.json',encoding="utf8")
intents_data = json.load(f)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        #no activation and no softmax
        return out

# load data of Pretrained Model
FILE = 'app/bot_model.pth'
data = torch.load(FILE)

# importing the hyperparameters
input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

# initialize and load the model
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# give a bot name
bot_name = 'kivous'

# define the function which will return responses
def get_response(msg):
    #Prepare the message data
    msg_tokens = tokenize(msg)
    msg_tokens = [stem(w) for w in msg_tokens]
    # X = bag_of_words(sentence, all_words)
    X = tf_idf(msg_tokens, all_words)

    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    # Classify message to one of the tags
    output= model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # probability of being in that class
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # define statements in case the classification probabilty is too low
    error_text= [
        "I'm sorry, I don't understand (._. )>",
        "I only could understand a little (⊙_⊙)?", 
        "Dont know that yet (。_。)", 
        "sorry could not understand ¯\(°_o)/¯",
        "I'm confused... (⊙ˍ⊙)"
    ]

    # return a random response 
    if prob.item() > 0.75:
        for intent in intents_data['intents']:
            if tag == intent['tag']:
                return random.choice(intent['responses']), prob.item(), tag
    else:
        return random.choice(error_text), prob.item(), tag
