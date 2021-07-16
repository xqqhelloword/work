import pandas as pd
from constant.special_string import *
import re
import jieba
import random
from tensorflow.keras.layers import Bidirectional, Dropout, BatchNormalization, Input, LSTM, Dense
from tensorflow.keras import Model
from intro_block_classification.crf import CRF


one_hot = {'O': 0, 'S': 1, 'J-B': 2, 'J-I': 3}


def stopwordslist():
    stopwords = [line.strip() for line in open(STOP_LIST_PATH, encoding='UTF-8').readlines()]
    return stopwords


def load_job(file_path):
    job_info = pd.DataFrame(columns=['job_direction', 'job_list'])
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        text = re.sub(r'\n', ' ', text)
    text_list = text.split('####')
    # print(text_list)
    for text in text_list:
        text = text.split(' ')
        if text[0] == '' or text[0] == ' ':
            job_info = job_info.append({'job_direction': text[1], 'job_list': text[2:]}, ignore_index=True)
        else:
            job_info = job_info.append({'job_direction': text[0], 'job_list': text[1:]}, ignore_index=True)
    # print(job_info)
    return job_info


def freq_analyze(job_info):
    dictionary = {}  # 装载所有的job单词
    word_list = []
    stop_words = stopwordslist()
    stop_words.append(' ')
    jieba.load_userdict(USER_DIC_PATH)
    for i in range(job_info.shape[0]):
        text = ' '.join(job_info.iloc[i, 1])
        text = re.sub(DEL_SPECIAL_PAT, '', text)
        text_list = jieba.lcut(text)
        text_list = [text for text in text_list if text != ' ']
        job_info.iloc[i, 1] = text_list
    count = 0
    for i in range(job_info.shape[0]):
        text_list = job_info.iloc[i, 1]
        for word in text_list:
            if word in dictionary.keys():
                dictionary[word][1] += 1  # 第二项为频率，第一项为索引
            else:
                word_list.append(word)
                dictionary[word] = [count, 1]
                count += 1
    bitmap = [[] for i in range(1000)]
    freq = []
    for key in dictionary.keys():
        bitmap[dictionary[key][1]].append(key)
    for i in range(len(bitmap) - 1, -1, -1):
        if bitmap[i]:
            freq.append([bitmap[i], i])
    print(dictionary)
    print(word_list)
    print('--------------------------------------------------------------')
    for f in freq:
        print(f)


def create_job_sentence(job_list, sentences, labels):
    """
    标签分为4种：O, S, J-B, J-I
    :param job_list: [job1, job2, ..., job_n]
    :param sentences: [ [word1, word2,...,word_m],  [word1, word2,...,word_m], ..., [word1, word2,...,word_m] ]
    :param labels: [ [l1, l2, ..., lm], [l1, l2, ..., lm], ... , [l1, l2, ..., lm] ]
    :return:
    """
    key = ['应聘职位', '应聘', '意向', '岗位', '意向岗位为', '意向岗位', '意向为', '应聘职位为',
           '担任过', '担任', '任职', '目标岗位', '目标岗位为', '职位', '职位目标是', '职业方向是', '曾担任过',
           '职位是', '岗位是', '想找', '工作目标岗位', '工作岗位', '工作', '工作是', '工作岗位为', '工作岗位是', '求职经历', '求职目标',
           '实习过', '做过', '想做', '胜任', '准备找', '准备寻求', '寻求', '准备', '/', '/', '/', '专业', '方向', '专业是', '方向是', '', '', ''
           ,'', '', '','', '', '','', '', '']
    characters = [' ', '-', '——', ':', '：', '', '']
    describe = ['分享', '介绍', '推荐', '建议', '发现', '觉得', '有过', '担任过', '经历过', '担任', '一年', '两年', '三年的', '四年', '五年的', '六年',
                '七年', '八年的', '九年', '十年的', '一个月', '两个月', '三个月', '半年', '四个月的', '五个月的', '六个月', '七个月的', '八个月', '九个月的'
        ,'十个月的', '11个月的', '1年', '2年', '3年', '4年', '5年', '6年的', '7年的', '8年', '9年', '10年','公司',
                '企业', '招聘', '/', '/', '', '', '', '', '', '', '', '', '', '']
    describe2 = ['工作经验', '的工作', '很不错', '是一个很好的职位', '这个职位', '的实习经验', '实习经历', '求职经历', '面试经历', '经验',
                 '经历', '工作介绍', '工作经历', '实习经验', '一年', '部门', '方向', '岗位', '相关', '内容', '领域', '工作期间', '前景很好', '待遇', '薪资待遇',
                 '岗位描述', '岗位职责', '的职责', '的薪资待遇', '的内容', '的技术栈', '的待遇', '/', '/', '/', '/', '', '', '', '', '', '', '', '','', '', '']

    # template1: key[i] + characters[i] + job_list[i] 800条
    # template2: describe[i]+ job_list[i] + describe2[i] 800条
    # template3 job_list[i] 500条
    # template4 no job 500条
    jieba.load_userdict(USER_DIC_PATH)
    for i in range(800):
        sen = []
        label = []
        d1 = random.choice(key)
        d1 = jieba.lcut(d1)
        for d in d1:
            sen.append(d)
            label.append('O')
        sen.append(random.choice(characters))
        label.append('O')
        job = random.choice(job_list)
        job = jieba.lcut(job)
        if len(job) == 1:
            sen.append(job[0])
            label.append('S')
        else:
            for j, word in enumerate(job):
                sen.append(word)
                if j == 0:
                    label.append('J-B')
                else:
                    label.append('J-I')
        if len(sen)!=0:
            sentences.append(sen)
            labels.append(label)

    for i in range(800):
        sen = []
        label = []
        d1 = random.choice(describe)
        job = random.choice(job_list)
        d2 = random.choice(describe2)
        d1 = jieba.lcut(d1)
        for d in d1:
            sen.append(d)
            label.append('O')
        job = jieba.lcut(job)
        if len(job) == 1:
            sen.append(job[0])
            label.append('S')
        else:
            for j, word in enumerate(job):
                sen.append(word)
                if j == 0:
                    label.append('J-B')
                else:
                    label.append('J-I')
        d2 = jieba.lcut(d2)
        for d in d2:
            sen.append(d)
            label.append('O')
        if len(sen)!=0:
            sentences.append(sen)
            labels.append(label)

    for i in range(500):
        sen = []
        label = []
        job = random.choice(job_list)
        job = jieba.lcut(job)
        if len(job) == 1:
            sen.append(job[0])
            label.append('S')
        else:
            for j, word in enumerate(job):
                sen.append(word)
                if j == 0:
                    label.append('J-B')
                else:
                    label.append('J-I')
        if len(sen)!=0:
            sentences.append(sen)
            labels.append(label)

    # for i, sentence in enumerate(sentences):
        # print(sentence)
    #     print(labels[i])
    #     print('--')


def one_hot_label(label):
    num = one_hot[label]
    res = []
    for i in range(4):
        if i == num:
            res.append(1)
        else:
            res.append(0)
    return res


def word2feature(word2vec, sentences, labels):
    """
    转为vector
    :param word2vec:
    :param sentences:
    :param labels:
    :return:
    """
    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            try:
                sentences[i][j] = word2vec[word]
            except KeyError:
                print('word2vec keyError:', word)
                # print(sentence)
                sentences[i][j] = word2vec['UAD']  # 用特殊字符代替未出现的词
                # print(sentences[i][j])
                labels[i][j] = one_hot_label('O')
                continue
            labels[i][j] = one_hot_label(labels[i][j])


def txt2feature_predict(word2vec, texts):
    return word2feature_predict(word2vec, texts)


def get_tokenize(texts):
    """

    :param texts: 未分词的文字
    :return: 分词后的文字
    """
    import copy
    tokenizes = copy.deepcopy(texts)
    jieba.load_userdict(USER_DIC_PATH)
    for i, text in enumerate(tokenizes):
        text = re.sub(DEL_SPECIAL_PAT2, '', text)
        tokenizes[i] = jieba.lcut(text)
    return tokenizes


def word2feature_predict(word2vec, tokenizes):
    """

    :param word2vec:
    :param tokenizes: 文字
    :return: 向量
    """
    import copy
    vec_arr = copy.deepcopy(tokenizes)
    for i, tokenize in enumerate(vec_arr):
        for j, word in enumerate(tokenize):
            try:
                vec_arr[i][j] = word2vec[word]
            except KeyError:
                print('word2vec keyError:', word)
                # sentences[i][j] = np.array([0 for i in range(100)], dtype=np.float)
                vec_arr[i][j] = word2vec['UAD']  # 用特殊字符代替未出现的词
    return vec_arr


def create_model():
    """
    LSTM+CRF
    :return:
    """
    input_tensor = Input(shape=(8, 100), name='input')
    x1 = Bidirectional(LSTM(units=100, activation='relu', return_sequences=True))(input_tensor)
    x1 = BatchNormalization()(x1)
    x2 = Bidirectional(LSTM(units=100, activation='relu', return_sequences=True))(x1)
    x2 = BatchNormalization()(x2)
    x2 = Dense(units=64, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)
    logit = Dense(units=4)(x2)
    crf = CRF(units=4, name='crf_layer')
    output = crf(logit)
    return Model(inputs=input_tensor, outputs=output), crf
