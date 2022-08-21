import nltk, json
nltk.download('punkt')
import numpy as np
from nltk.stem.porter import PorterStemmer

# read data
f = open('app/bot_data.json',encoding="utf8")
intents_data = json.load(f)

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)

stemmer = PorterStemmer()
def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())

#prepare data
all_tokens = []
tags = []
xy = []
for intent in intents_data['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        tokens = tokenize(pattern)
        tokens = [stem(w) for w in tokens]
        all_tokens.extend(tokens)
        xy.append((tokens, tag))
     
ignore_words = ['.', ',']
all_tokens = [w for w in all_tokens if w not in ignore_words]
all_tokens = sorted(set(all_tokens))
tags = sorted(set(tags))

X_train = []
y_train = []

word_count = {}
for word in all_tokens:
    word_count[word] = 0
    for tokenized_sentence, _ in xy:
        if word in tokenized_sentence:
            word_count[word] += 1

def term_freq(word, sentence):
    # count of word in sentence 
    # number of words in sentence len(sentence)
    sentence_len = len(sentence)
    occurence = len([token for token in sentence if token == word])
    return occurence/sentence_len
   
def tf_idf(pattern_tokens, all_tokens):
    """ return tf-idf array:
        1 for each known word that exists in the sentence, 0 otherwise
        example:
        pattern_tokens = ["hello", "how", "are", "you"]
        all_tokens = ["hi", "hello", "I", "you", "bye", "thank", "cool", "are"]
        vector    = [ 0  ,   0.8  ,  0 ,  0.6 ,   0  ,    0   ,    0  ,  0.7 ]
    """
    vector = np.zeros(len(all_tokens), dtype=np.float32)
    
    for idx, token in enumerate(all_tokens): 
        if token in pattern_tokens: 
            idf = np.log(len(xy) / word_count[token])
            tf = term_freq(token, pattern_tokens)
            vector[idx] = idf * tf
    return vector


# previous bag-of-words model
# def bag_of_words(tokenized_sentence, all_words):
    # """
    # return bag of words array:
    # 1 for each known word that exists in the sentence, 0 otherwise
    # example:
    # sentence_words = ["hello", "how", "are", "you"]
    # all_words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    # bag       = [ 0  ,    1   ,  0 ,   1  ,   0  ,    0   ,    0  ]
    # """
    # sentence_words = [stem(w) for w in tokenized_sentence]
    
    # bag = np.zeros(len(all_words), dtype=np.float32)
    # for idx, word in enumerate(all_words):
    #     if word in sentence_words:
    #         bag[idx] = 1
    
    # return bag
