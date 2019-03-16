import numpy as np
import spacy
from keras.preprocessing import sequence

# initialization

null_class = 'O'
len_word_vectors = 60
len_named_class = 8
nlp = spacy.load('en_core_web_sm')

tags = []
tags_map = {}
tags_one_hot_map = {}

def process_tag(tags):
    """ return a processed tags """
    return tags[2:]

def get_word_vector(word):
    """ returns a vector of a word using GloVe model"""
    s = nlp(word)
    return s.vector

def read_data(datapath):
    
    """ reading in the data line by line """
    _id = 0
    sentence = []
    sentence_tag = []
    x, y = [], []
    max_len = 0
    all_data = []
    
    with open(datapath, 'r') as f:
        for l in f:
            line = l.strip().split()
            if line:
                word, named_tags = line[0], line[1]
                if named_tags != null_class:
                    named_tags = process_tag(named_tags)
                    
                if named_tags not in tags:
                    tags.append(named_tags)
                    tags_map[_id] = named_tags
                    one_hot_vec = np.zeros(len_named_class, dtype = np.int32)
                    one_hot_vec[_id] = 1
                    print(one_hot_vec)
                    tags_one_hot_map[named_tags] = one_hot_vec
                    
                    _id += 1
                    
                sentence.append(get_word_vector(word)[:len_word_vectors])
                sentence_tag.append(tags_one_hot_map[named_tags])
			
            else:
                all_data.append((sentence, sentence_tag))
                sentence = []
                sentence_tag = []
        
    all_data.append((sentence, sentence_tag))

    # sentence max length
    for pair in all_data:
        if max_len < len(pair[0]):
            max_len = len(pair[0])
#            print(len(pair[0]), max_len)
                    
    for vectors, one_hot_tags in all_data:
        temp_x = np.zeros(len_word_vectors, dtype = np.int32) 
        temp_y = np.array(tags_one_hot_map[null_class])
        pad_length = max_len - len(vectors)
        
        x.append((pad_length * [temp_x]) + vectors)
        y.append((pad_length * [temp_y]) + one_hot_tags)
        
    X, Y = np.array(x), np.array(y)
    return X, Y, max_len

x, y, maxLen = read_data("./data/sample_text_manual.txt")
     
def encode_sentence(sentence):
    vectors = get_score_sentence(sentence)
    vectors = [v[:len_word_vectors] for v in vectors]
    return sequence.pad_sequences([vectors], maxlen = maxLen, dtype = np.float32)

def get_score_sentence(sentence):
    """returns a vectors for a sentence as list"""
    sent = sentence.strip().split()
    vec = [get_word_vector(word) for word in sent]
    return vec  
    
def decode_result(pred_results):
    pred_tags = []
    for tags in pred_results[0]:
        _id = np.argmax(tags)
        pred_tags.append(tags_map[_id])
    return pred_tags