import numpy as np
import spacy
from keras.preprocessing import sequence

# initialization

null_class = 'O'
len_word_vectors = 60
len_named_class = 5
nlp = spacy.load('en_core_web_sm')


tags = []
tags_map = {}
tags_one_hot_map = {}

def read_data(datapath):
    
    """ reading in the data line by line """
    _id = 0
    sentence = []
    sentence_tag = []
    all_data = []
    x, y = [], []
    max_len = 0
    
    with open(datapath, 'r') as f:
        for l in f:
            line = l.strip().split()
            if line:
                word, named_tags = line[0], line[3]
                if named_tags != null_class:
                    named_tags = process_tag(named_tags)
                    
                if named_tags not in tags:
                    tags.append(named_tags)
                    tags_map[_id] = named_tags
                    one_hot_vec = np.zeros(len_named_class, dtype = np.int32)
                    one_hot_vec[_id] = 1
                    tags_one_hot_map[named_tags] = one_hot_vec
                    
                    _id += 1
                    
                sentence.append(get_word_vector(word)[:len_word_vectors])
                sentence_tag.append(tags_one_hot_map[named_tags])
                print(len(sentence))
			
            else:
                all_data.append((sentence, sentence_tag))
                print("alldata length :" + len(all_data))
                sentence = []
                sentence_tag = []
                
    return all_data
           
        
    
#    # sentence max length
#    for pair in all_data:
#        if max_len < len(pair[0]):
#            max_len = len(pair[0])
#                    
#    for vectors, one_hot_tags in all_data:
#        temp_x = np.zeros(len_named_class, dtype = np.int32) 
#        temp_y = np.array(tags_one_hot_map[null_class])
#        pad_length = max_len - len(vectors)
#        
#        x.append((pad_length * [temp_x]) + vectors)
#        y.append((pad_length * [temp_y]) + one_hot_tags)
#        
#    x, y = np.array(x), np.array(y)
#    return x, y, max_len
                
                
                
                
                
def encode_sentence(sentence):
    vectors = get_score_sentence(sentence)
    vectors = [v[:len_word_vectors] for v in vectors]
    return sequence.pad_sequences([vectors], maxlen = 13, dtype = np.float32)

def process_tag(tags):
    """ return a processed tags """
    return tags[2:]

def get_score_sentence(sentence):
    """returns a vectors for a sentence as list"""
    sent = sentence.strip().split()
    vec = [get_word_vector(word) for word in sent]
    return vec  
    
def get_word_vector(word):
    """ returns a vector of a word using GloVe model"""
    s = nlp(str(word))
    return s.vector
              
ad = read_data("../data/eng_data.txt")

