from embedding import WordEmbedding
import numpy as np
import pickle
from gensim.models import LdaModel

# calculate the topic centrality by keywords and their weights
def topic_keyword_mapping(corpus_name, topic, topic_relevancy_map, wembedder):
    topic_kws_map = {topic:{}, "other": {}}
    kw_topic_map = {}
    kw_vectors = {}
    lda = LdaModel.load("lda_model/"+corpus_name+"_"+topic+"_lda_c10_model")
    topic_keywords = lda.print_topics(20, num_words=60)
    for topic in topic_keywords:
        tname = topic_relevancy_map[topic[0]]
        print(tname)
        kws = topic[1].split("+")
        for kw in kws:
            kw_factors = kw.strip()[:-1].split('*"')
            if len(kw_factors) < 2:
                continue
            kw_c = kw_factors[1].strip()
            if kw_c not in topic_kws_map[tname]:
                topic_kws_map[tname][kw_c] = []
            try:
                topic_kws_map[tname][kw_c].append(float(kw_factors[0].strip()))
            except:
                print(kw_c, topic_kws_map[tname])
            if kw_factors[1] not in kw_topic_map:
                kw_topic_map[kw_factors[1]] = []
                kw_vectors[kw_factors[1]] = wembedder.get_word_vector(kw_factors[1])
            kw_topic_map[kw_factors[1]].append((tname, kw_factors[0]))
    for tname in topic_kws_map:
        topic_kws_map[tname] = {kw_c: np.mean(np.array(kw_weights)) for kw_c, kw_weights in topic_kws_map[tname].items()}
        
        
    return topic_kws_map, kw_vectors, kw_topic_map


def topic_centralization_calc(topic_kws_map, wembedder):
    # Before calculate the centrality, normalization the weight
    # Two norm approach:
    ## a1: sum(weight)=1 then mean is weight*vector
    ## a2: mean(weight)=1, then mean(weight*vector)
    topic_centralities = {}
    for tname, kwws in topic_kws_map.items():
        kwvs = np.array([wembedder.get_word_vector(kw) for kw in kwws.keys()])
        weight = np.array([kwws[kw] for kw in kwws])
        print(kwvs.shape, weight.shape)
        weight_norm = weight/weight.sum() # a1
        centrality = weight_norm.dot(kwvs)
        print(centrality.shape)
        topic_centralities[tname] = centrality
        
    return topic_centralities
        

def distance(vec1, vec2):
    # calucate the weight of each word in a topic
    # weight(t1)* dist(t1)-sum(weight(tj)*dist(tj))
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return dist


def keywords_ranking(topic, kw_topic_map, topic_centralities, kw_vectors, topic_kw_file,topic_kws_map):
    topic_kw_weight = {topic:{}, "other": {}}
    for tname, kwws in topic_kws_map.items():
        for kww in kwws:
            including_tid = kw_topic_map[kww]
            tid_weight = kwws[kww]
            dist_tid = distance(topic_centralities[tname], kw_vectors[kww])
            weight = float(tid_weight)/max(dist_tid, 0.01)
            for other_tid, other_weight in kw_topic_map[kww]:
                if other_tid==tname:
                    continue
                dist_other = distance(topic_centralities[other_tid], kw_vectors[kww])
                weight -= (float(other_weight) / max(0.01, dist_other))
            topic_kw_weight[tname][kww] = weight
    kw_sorted = sorted(topic_kw_weight[topic].items(), key=lambda x:x[1], reverse=True)
    print(kw_sorted[:70])
    with open(topic_kw_file, "wb") as f:
        pickle.dump(kw_sorted[:70], f)