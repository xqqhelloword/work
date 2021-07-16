import spacy
import tensorflow as tf
import numpy as np
from intro_block_classification.share_func import *
import gensim
np.set_printoptions(suppress=True)

spacy.prefer_gpu()
nlp = spacy.load('zh_core_web_lg')

# -----------------------------------------------------教育背景---------------------------------------- #


def get_entity(pre_res, tokenizes):
    """

    :param pre_res: [[0,1,2,2,3,0],[...],[...]]
    :param tokenizes: [[word1, word2, ...],[word1, word2, ...],[word1, word2,...]]
    :return:
    """
    entities = []
    for i, pre in enumerate(pre_res):
        tokenize = tokenizes[i]
        # print(tokenize)
        # print(pre)
        ent = []
        j = 0
        length = min(len(tokenize), 8)
        while j < length:
            if pre[j] == 1:  # s
                ent.append(tokenize[j])
                j += 1
            elif pre[j] == 2:
                str1 = tokenize[j]
                j += 1
                while j < length and pre[j] == 3:
                    str1 += tokenize[j]
                    j += 1
                ent.append(str1)
            else:
                j += 1
        entities.append(ent)
    return entities


def extract_item_edu(texts, nlp, job_model, dic, word2vec):
    """
    提取教育背景中的内容项：学校、专业、学历、就读时间
    学校名称、专业名称：ner名称提取
    学历、时间：规则提取
    :param texts:句子列表
    :param nlp:
    :param job_model:识别专业
    :param word2vec:txt2vec模型
    :return:
    """
    for i, text in enumerate(texts):
        doc = nlp(text)
        # show_ents(doc)
        # print('-------------------------------------------------')
        if doc.ents:
            for ent in doc.ents:
                if ent.label_ == 'ORG':  # 可能是学校名、公司名
                    dic['school'].append(ent.text)
                else:
                    pass
    texts = [text for text in texts if 16 > len(text) > 2 and re.search('[0-9]+', text) is None]
    tokenizes = get_tokenize(texts)
    x = word2feature_predict(word2vec, tokenizes)
    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=8, padding='post', dtype='float',
                                                              truncating='post')
    x = np.array(x)
    pre_y = job_model.predict(x)
    res = get_entity(pre_y, tokenizes)
    for jobs in res:
        for job in jobs:
            if job not in dic['school']:
                dic['major'].append(job)


def extract_item_edu_by_rule(texts, dic):
    """
    通过规则提取学历以及就读时间
    :param texts:
    :param dic:
    :return:None
    """
    for i, text in enumerate(texts):
        res = re.search(r"(\d{4}[ ]{0,3}/[ ]{0,3}\d{1,2}[ ]{0,3}"
                        r"/?[ ]{0,3}\d{0,2}[ ]{0,3})[ ]{0,3}[-到至]{0,3}[ ]{0,3}(\d{4}[ ]{0,3}"
                        r"/[ ]{0,5}\d{1,2}[ ]{0,3}/?[ ]{0,3}\d{0,2}[ ]{0,3})?", text)
        if res is None:
            res = re.search(r"(\d{4}[ ]{0,3}\.[ ]{0,3}\d{1,2}[ ]{0,3}"
                        r"\.?[ ]{0,3}\d{0,2}[ ]{0,3})[ ]{0,3}[-到至]{0,3}[ ]{0,3}(\d{4}[ ]{0,3}"
                        r"\.[ ]{0,5}\d{1,2}[ ]{0,3}\.?[ ]{0,3}\d{0,2}[ ]{0,3})?", text)
            if res is None:
                res = re.search(r"(\d{4}[ ]{0,3}-[ ]{0,3}\d{1,2}[ ]{0,3}"
                        r"-?[ ]{0,3}\d{0,2}[ ]{0,3})[ ]{0,3}[-到至]{0,3}[ ]{0,3}(\d{4}[ ]{0,3}"
                        r"-[ ]{0,5}\d{1,2}[ ]{0,3}-?[ ]{0,3}\d{0,2}[ ]{0,3})?", text)
        if res is not None:
            start = res.group(1)
            end = res.group(2)
            if start is not None:
                dic['start_time'].append(start)
            else:
                dic['start_time'].append('null')
            if end is not None:
                dic['end_time'].append(end)
            else:
                dic['end_time'].append('null')
        res = re.search(r'(学[ ]{0,10}位|学[ ]{0,10}历|学[ ]{0,10}历\\学[ ]{0,10}位)?'
                        r'[:：\-]?(小[ ]{0,10}学|初[ ]{0,10}中|高[ ]{0,10}中|专[ ]{0,10}科|本[ ]{0,10}'
                        r'科|学[ ]{0,10}士|研[ ]{0,10}究[ ]{0,10}生|硕[ ]{0,10}士|博[ ]{0,10}士)', text)
        if res is not None:
            # print('学历:', res.group(2))
            dic['scholar'].append(res.group(2))


def test_extract_edu():
    texts = [
        '2013.06.21 - 2017.5',
        '康奈尔大学',
        '硕士',
        '方向：数据分析与机器学习'
        '2011 / 9 - 2013 / 6',
        '浙江大学',
        '本科',
        '专业：计算机科学与技术'
    ]
    dic = {'school': [], 'start_time': [], 'end_time': [], 'major': [], 'scholar': []}
    model, _ = create_model()
    model.load_weights(JOB_NAME_EXTRACTION_MODEL_PATH)
    word2vec = gensim.models.Word2Vec.load(word2vec_model_path_2021_4_2)
    extract_item_edu(texts=texts, nlp=nlp, dic=dic, word2vec=word2vec, job_model=model)
    extract_item_edu_by_rule(texts, dic)
    print(dic)
