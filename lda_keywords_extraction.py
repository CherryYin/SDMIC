import pandas as pd
import os
import re
import logging
import tqdm
import nltk
# from pattern.en import lemma
from collections import Counter
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import jieba.posseg
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sentence_process(sentences, stop_words, lang="en"):
    if lang == "en":
        valid_pos = {"NN", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
    else:
        valid_pos = {"n", "nr", "ns", "nsf", "nt", "nz"}
    words_list = []
    invalid_indexes = []
    for i in range(len(sentences)):
        s = sentences[i]
        if lang == "en": 
            s = s.replace(",", " , ").replace(".", " . ")
            s = s.replace("?", " . ").replace("!", " . ")
            s = s.replace("\\", "")
            words = [w.strip() for w in s.lower().split(" ") if w not in stop_words]
            words = [w for w in words if re.search("^a-z", w)==None and w!=""]
            if len(words) < 4:
                invalid_indexes.append(i)
                continue
            # words = [w for w, pos in nltk.pos_tag(words) if pos in valid_pos]
        else:
            words = [w for w, pos in jieba.posseg.cut(s.lower()) if w not in stop_words and pos in valid_pos]
        
        """if len(words)==0:
            continue
        try:
            words = [lemma(w) for w in words]
            words = [w[:-1] if w[-1]=="'" else w for w in words]
        except:
            print("**", words)"""
        words_list.append(words)
    return words_list, invalid_indexes

def vocab_generation(words_list):
    total_words = []
    for words in words_list:
        total_words.extend(words)
    words_map = Counter(total_words)
    del total_words
    vocab = [w for w, cnt in words_map.items() if cnt>30]
    
    return set(vocab)


def feature_generation(stop_words, data_path, lang="en"):
    dirs = os.listdir(data_path)
    sentences, labels = [], []
    for dirr in dirs:
        if dirr.find("checkpoints")!=-1:
            continue
        file = os.path.join(data_path, dirr)
        with open(file, "r", encoding="utf8") as f:
            sentences.extend([l.strip() for l in f.readlines()])
            labels.extend([1 for i in range(30)])
            labels.extend([0 for i in range(20)])
    assert len(sentences)==len(labels) 
    

    logger.info("Total sentence number is {}.".format(len(sentences)))
    logger.info("sentence split and word process finished!")
    words_list, invalid_indexes = sentence_process(sentences, stop_words, lang)
    sentences = [sentences[i] for i in range(len(sentences)) if i not in invalid_indexes]
    labels = [labels[i] for i in range(len(labels)) if i not in invalid_indexes]
    df = pd.DataFrame()
    df["sentence"] = sentences
    df["lable"] = labels
    logger.info("Valid sentence number is {}.".format(len(sentences)))
    logger.info("Word map have been created!")
    # vocab = vocab_generation(words_list)
    # logger.info("Totally {} words!".format(len(vocab)))
    
    return words_list, df
    
    
def lda_training(words_list, corpus_name, topic, num_topics=8):
    dictionary = Dictionary(words_list)
    corpus = [ dictionary.doc2bow(text) for text in words_list ]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    topics = lda.get_document_topics(corpus)
    
    lda.save("lda_model/"+corpus_name+"_"+topic+"_lda_c10_model")
    with open("lda_model/dictionary_"+corpus_name+"_"+topic+".pkl", "wb") as f:
        pickle.dump(dictionary, f)
    
    return lda


def lda_predict(words_list, corpus_name, topic, num_topics=8):
    print("lda_model/dictionary_"+corpus_name+"_"+topic+".pkl")
    with open("lda_model/dictionary_"+corpus_name+"_"+topic+".pkl", "rb") as f:
        dictionary = pickle.load(f)
    corpus = [dictionary.doc2bow(text) for text in words_list]
    # lda = LdaModel.load("lda_model/"+corpus_name+"_"+topic+"_lda_c10_model")
    lda = LdaModel.load("lda_model/"+topic+"_lda_c10_model")
    topics = lda.get_document_topics(corpus)
    
    topic_ids = []
    for i in range(len(topics)):
        topic_ids.append(sorted(topics[i], key=lambda x:x[1], reverse=True)[0][0])
    return topic_ids
    
    


