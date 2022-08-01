


topic_ids = []
for i in range(len(topics)):
    topic_ids.append(sorted(topics[i], key=lambda x:x[1], reverse=True)[0][0])
df["topic"]=topic_ids
df.to_excel("business_topic_result.xlsx")

data_path = "data/AG_news/bags/Business/test/positive"
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
test_df = pd.DataFrame()
test_df["sentence"] = sentences
test_df["lable"] = labels

logger.info("Total sentence number is {}.".format(len(sentences)))
logger.info("sentence split and word process finished!")
test_words_list = sentence_process(sentences, stop_words)
logger.info("Valid sentence number is {}.".format(len(sentences)))
logger.info("Word map have been created!")
corpus = [ dictionary.doc2bow(text) for text in test_words_list ]
test_topics = lda.get_document_topics(corpus)

test_topics_ids = []
for i in range(len(test_topics)):
    test_topics_ids.append(sorted(test_topics[i], key=lambda x:x[1], reverse=True)[0][0])
test_df["topic"] = test_topics_ids
test_df.to_excel("business_topic_test_result.xlsx")