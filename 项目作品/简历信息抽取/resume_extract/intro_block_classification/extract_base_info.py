# coding=utf-8

"""
@File  : extract_base_info.py
@Author: Xu Qiqiang
@Date  : 2021/1/27 0027
本模块为简历块内NER的API，主要包括基本信息、教育背景、工作经历提取、荣誉证书提取、论文/专利提取、技术栈提取
"""
import spacy
import gensim
import numpy as np
from resume_block_classification.data_precess.load_resume_data import stopwordslist
from intro_block_classification.share_func import *
import tensorflow as tf
np.set_printoptions(suppress=True)
spacy.prefer_gpu()

# nlp = spacy.load('en_core_web_sm')  # Write a function to display basic entity info:
nlp = spacy.load('zh_core_web_lg')

one_hot = {'O': 0, 'S': 1, 'J-B': 2, 'J-I': 3}


def corpus_processing(file_path):

    with open(file_path, 'r',encoding='utf-8') as f:
        text = f.read()
        texts = text.split('********************************************************************')
        for i, t in enumerate(texts):
            texts[i] = jieba.lcut(re.sub(DEL_SPECIAL_PAT1, '', t))

    with open(file_path,'w', encoding='utf-8') as f:
        for words in texts:
            for word in words:
                if word != '\n':
                    f.write(word)
                    f.write('\n')
            f.write('###\n')


def corpus_processing1(file_path):
    texts1 = []
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
        for i, text in enumerate(texts):
            text = re.sub(r'\(.*?\)', '', text)
            if len(text) != 0 or text != ' ' or text != '\n':
                print(text)
                texts1.append(text)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(texts1)


def seg(text, split):
    i=0
    while i < len(text) and text[i] not in split:
        i += 1
    if i+1 < len(text):
        return text[:i+1], text[i+1:]
    return text, ''

# -----------------------------------------------------基本信息---------------------------------------- #


def extract_item_base_info(texts, nlp, dic):
    """
    提取基本信息中的内容项：姓名、性别、年龄、工作时长、住址、籍贯、出生年月、联系方式、毕业院校、学历、政治面貌、职业方向、个人链接
    :param texts:句子列表
    :param nlp:
    :return:
    """
    for i, text in enumerate(texts):
        doc = nlp(text)
        # show_ents(doc)
        # print('-------------------------------------------------')
        if doc.ents:
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    res = re.search(CORRECT_NAME, ent.text)
                    if res is not None:
                        # print(res)
                        continue
                    dic['name'] .append(ent.text)
                elif ent.label_ == 'ORG':  # 可能是学校名、公司名、网址形式（需要修正）
                    res = re.search(CORRECT_LINK, ent.text)
                    res1 = re.search(CORRECT_LINK1, ent.text)
                    if res is not None or res1 is not None:
                        pass
                    else:
                        dic['school'].append(ent.text)
                elif ent.label_ == 'DATE' or ent.label_ == 'CARDINAL':  # 可能是长整型时间字符串（造成数字类型误判为日期）、规则化时间字符串
                    if len(ent.text) in [6, 7, 8, 11, 12, 13, 14] and re.search(r'[^\d\-+]', ent.text) is None:
                        # 如果数字长度符合该范围以及仅仅包含指定数字和字符，则为联系方式
                        # dic['contact'].append(ent.text)
                        pass
                    elif re.search('年[ ]{0,6}龄|岁|手[ ]{0,6}机|联[ ]{0,6}系[ ]{0,6}方[ ]{0,6}式|'
                                   '电[ ]{0,6}话[ ]{0,6}号[ ]{0,6}码|号[ ]{0,6}码|'
                                   '电[ ]{0,6}话|周[ ]{0,6}岁|岁[ ]{0,6}数|Q[ ]{0,6}Q|微[ ]{0,6}信', text) is not None:
                        # item = re.sub(':|：|年龄|岁|岁数', '', ent.text)
                        # dic['age'].append(item)
                        pass
                    elif i > 0:
                        if re.search('手[ ]{0,10}机|联[ ]{0,10}系[ ]{0,10}方[ ]{0,10}式|'
                                     '电[ ]{0,10}话[ ]{0,10}号[ ]{0,10}码|号[ ]{0,10}码|电[ ]{0,10}话|年[ ]{0,10}龄|'
                                     '岁[ ]{0,10}数|QQ|微[ ]{0,10}信', texts[i-1], re.IGNORECASE) is not None:
                            # dic['contact'].append(ent.text)
                            pass
                        elif re.search('年龄|岁数', texts[i-1]) is not None:
                            # item = re.sub(':|：|年龄|岁|岁数', '', ent.text)
                            # dic['age'].append(item)
                            pass
                        else:
                            dic['work_time'] .append(ent.text)
                    else:
                        dic['work_time'] .append(ent.text)

                elif ent.label_ == 'GPE':  # 可能是住址、籍贯。用规则加以区分：如果有前缀提示，往前搜索字符“地址，住址...”、“籍贯、户口、国籍等”
                    if i == 0:
                        res = re.search('住[ ]{0,10}址|地[ ]{0,10}址|现[ ]{0,10}居[ ]{0,10}住[ ]{0,10}'
                                        '地|居[ ]{0,10}住[ ]{0,10}地|居[ ]{0,10}住[ ]{0,10}地[ ]{0,10}址|'
                                        '所[ ]{0,10}在[ ]{0,10}地|工[ ]{0,10}作[ ]{0,10}单[ ]{0,10}位|'
                                        '工[ ]{0,10}作[ ]{0,10}单[ ]{0,10}位[ ]{0,10}地[ ]{0,10}址|'
                                        '常[ ]{0,10}居[ ]{0,10}住[ ]{0,10}地|常[ ]{0,10}住[ ]{0,10}地[ ]{0,10}址|'
                                        '单[ ]{0,10}位[ ]{0,10}地[ ]{0,10}址', text)
                        if res is not None:
                            dic['place'].append(ent.text)
                        else:
                            dic['city'].append(ent.text)
                    else:
                        res = re.search('住[ ]{0,10}址|地[ ]{0,10}址|现[ ]{0,10}居[ ]{0,10}住[ ]{0,10}'
                                        '地|居[ ]{0,10}住[ ]{0,10}地|居[ ]{0,10}住[ ]{0,10}地[ ]{0,10}址|'
                                        '所[ ]{0,10}在[ ]{0,10}地|工[ ]{0,10}作[ ]{0,10}单[ ]{0,10}位|'
                                        '工[ ]{0,10}作[ ]{0,10}单[ ]{0,10}位[ ]{0,10}地[ ]{0,10}址|'
                                        '常[ ]{0,10}居[ ]{0,10}住[ ]{0,10}地|常[ ]{0,10}住[ ]{0,10}地[ ]{0,10}址|'
                                        '单[ ]{0,10}位[ ]{0,10}地[ ]{0,10}址', text)

                        res1 = re.search('住[ ]{0,10}址|地[ ]{0,10}址|现[ ]{0,10}居[ ]{0,10}住[ ]{0,10}'
                                        '地|居[ ]{0,10}住[ ]{0,10}地|居[ ]{0,10}住[ ]{0,10}地[ ]{0,10}址|'
                                        '所[ ]{0,10}在[ ]{0,10}地|工[ ]{0,10}作[ ]{0,10}单[ ]{0,10}位|'
                                        '工[ ]{0,10}作[ ]{0,10}单[ ]{0,10}位[ ]{0,10}地[ ]{0,10}址|'
                                        '常[ ]{0,10}居[ ]{0,10}住[ ]{0,10}地|常[ ]{0,10}住[ ]{0,10}地[ ]{0,10}址|'
                                        '单[ ]{0,10}位[ ]{0,10}地[ ]{0,10}址', texts[i-1])
                        if res is not None or res1 is not None:
                            dic['place'].append(ent.text)
                        else:
                            dic['city'].append(ent.text)
                else:
                    pass


def extract_item_by_rule_base_info(texts, dic):
    """
    提取 微信、QQ、手机号码、邮箱、性别、政治面貌、学历、年龄、个人链接
    :param texts:
    :return:
    """
    for i, text in enumerate(texts):
        # 检测邮箱
        res = re.search(r"[-_\w\.]{0,64}[ ]{0,5}@[ ]{0,5}([-\w]{1,63}\.)*[-\w]{1,63}", text)
        if res is not None:
            # print('email:', res.group())
            dic['contact'].append('邮箱:'+res.group())
        # 检测手机号
        res = re.search('[\d]{0,5}[+\-]?[\d]{7,11}', text)
        if res is not None:
            res1 = re.search('微[ ]{0,10}信|Q[ ]{0,10}Q|email', text, re.IGNORECASE)
            res2 = re.search('手[ ]{0,10}机|电[ ]{0,10}话|座[ ]{0,10}机|号[ ]{0,10}码', text)
            if res1 is None or res2 is not None:
                text1 = res.group()
                if re.search('[\-+]', text1) is not None:
                    dic['contact'].append('电话号码:' + res.group())
                else:
                    if len(text1) == 11:
                        dic['contact'].append('电话号码:' + res.group())
        # 检测微信
        res = re.search(r'(微[ ]{0,10}信[^a-zA-Z\d@#^&*%$!_-]*)[:：\- ]{0,2}([a-zA-Z\d@#^&*%$!_-]{3,})', text)
        if res is not None:
            # print(res.groups())
            # print('微信：', res.group())
            dic['contact'].append('微信:'+res.group(2))
        else:
            res = re.search(r'([a-zA-Z\d@#^&*%$!_-]{3,})([(（【{\[\-]微[ ]{0,10}信[)}）】\]]*)', text)
            if res is not None:
                # print('微信:', res.group(1))
                dic['contact'].append('微信:'+res.group(1))
        # 检测性别
        res = re.search('(性[ ]{0,10}别)?[:：\- ]?([男女]).*', text)
        if res is not None:
            # print('性别:', res.group(2))
            dic['gender'].append(res.group(2))
        # 检测政治面貌
        res = re.search('共[ ]{0,5}青[ ]{0,5}团[ ]{0,5}员|群[ ]{0,5}众|党[ ]{0,5}员|'
                        '中[ ]{0,5}共[ ]{0,5}党[ ]{0,5}员|普[ ]{0,5}通[ ]{0,5}公[ ]{0,5}民', text)
        if res is not None:
            # print('政治面貌：', res.group())
            dic['politic'].append(res.group())
        # 检测学历
        res = re.search(r'(学[ ]{0,10}位|学[ ]{0,10}历|学[ ]{0,10}历\\学[ ]{0,10}位)?'
                        r'[:：\-]?(小[ ]{0,10}学|初[ ]{0,10}中|高[ ]{0,10}中|专[ ]{0,10}科|本[ ]{0,10}'
                        r'科|学[ ]{0,10}士|研[ ]{0,10}究[ ]{0,10}生|硕[ ]{0,10}士|博[ ]{0,10}士)', text)
        if res is not None:
            # print('学历:', res.group(2))
            dic['scholar'].append(res.group(2))
        # 年龄
        res = re.search(r'(年[ ]{0,10}龄)[:\-： ]{0,3}([\d]{1,2})', text)
        if res is not None:
            # print(res.groups())
            # print(res.string)
            # print('年龄:', res.group(2))
            dic['age'].append(res.group(2))
        else:
            res = re.search(r'([\d]{1,2})(周[ ]{0,10}岁|岁)', text)
            if res is not None:
                # print(res.string)
                # print('年龄:', res.group(1))
                dic['age'].append(res.group(1))
        res = re.search(CORRECT_LINK, text)
        res1 = re.search(CORRECT_LINK1, text)
        if res is not None:
            print(res.groups())
            dic['link'].append(res.group())
        elif res1 is not None:
            print(res.groups())
            dic['link'].append(res1.group())


# -----------------------------------------------------基本信息(job extraction)---------------------------------------- #


def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text+' - ' + str(ent.start_char) + ' - ' + str(ent.end_char) + ' - ' + ent.label_ + ' - ' +
                  str(spacy.explain(ent.label_)))
    else:
        print('No named entities found.')


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


def test0():
    dic = {'name': [], 'gender': [], 'age': [], 'work_time': [], 'place': [], 'city': [], 'birth': [], 'politic': [],
           'contact': [], 'link': [], 'job': [], 'scholar': [], 'school': []}
    texts = ['联系方式：18170702941', 'Email:woeir@qq.com', '学  位\学 历:小学生', '政 治 面 貌——群众', '性 别：男', '微信xqq1226922778)',
             '毕业于杭州电子科技大学', '个人链接：https://www.xuqiqiang.com', '许启强', '籍贯\\户口：中国台湾', '工作单位：高雄市', '23岁']
    texts1 = [
        '应聘职位：', '浙江大学 本科', '硕士：清华大学', '周伟光', 'ID:653228941', '目前正在找工作', '17306810051', 'zhouweiguang@hotmail.com'
    ]
    texts2 = [
        '个人简历',
            '细心从每一个小细节开始。',
            'Personal resume,',
            '个人简历',
            '细心从每一个小细节开始。',
            'Personal resume',
            '周星驰',
            '民族：汉',
            '电话：18817636118',
            '邮箱：1792767532 @ qq.com',
            '住址：杭州市滨江区',
            '性别：男',
            '出生年月：1993.03.01',
            '身高：168cm',
            '学历：硕士研究生',
            '院校：上海大学（211）'
            ]
    extract_item_base_info(texts2, nlp, dic)
    extract_item_by_rule_base_info(texts2, dic)
    print(dic)


def test_create_word2vec_corpus():
    file_path = ROOT_PATH + '\\专业大全.txt'
    with open(file_path,'r', encoding='utf-8') as f:
        job_info = f.readlines()
    tokenizes = get_tokenize(job_info)
    for i, token in enumerate(tokenizes):
        for t in token:
            print(t, end=" ")
        if i % 20 == 0:
            print("")


# -----------------------------------------------------项目经历---------------------------------------- #
def test1():
    """
    用于添加语料，训练抽取模型
    :return:
    """

    file_path = ROOT_PATH + '\\job_order.txt'
    job_info = load_job(file_path)
    with open(ROOT_PATH + '\\专业大全.txt','r', encoding='utf-8') as f:
        job_list = f.readlines()
    for i in range(job_info.shape[0]):
        jobs = job_info.iloc[i, 1]
        for job in jobs:
            job = re.sub(DEL_SPECIAL_PAT2, '', job)
            job_list.append(job)
    sentences = []
    labels = []
    create_job_sentence(job_list, sentences=sentences, labels=labels)
    # # print(sentences[0])
    word2vec = gensim.models.Word2Vec.load(word2vec_model_path_2021_4_2)
    word2feature(word2vec=word2vec, sentences=sentences, labels=labels)
    model, crf = create_model()
    model.compile(optimizer='adam',
                  loss={'crf_layer': crf.get_loss},
                  metrics=['accuracy'])
    sentences = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=8, padding='post', dtype='float', truncating='post')
    train_x = np.array(sentences)
    labels = tf.keras.preprocessing.sequence.pad_sequences(labels, maxlen=8, padding='post',truncating='post')
    train_y = np.array(labels)
    model.fit(train_x[:1500], np.argmax(train_y[:1500], axis=-1), batch_size=32, epochs=10)
    scores = model.test(train_x[1500:], np.argmax(train_y[1500:], axis=-1), verbose=1)
    print(scores)
    test_text = ['求职意向——数学教师', '担任过销售主管这个职位','期待岗位为数据分析师','金融统计与风险管理方向（研究生）',
                 '浙江大学计算机系','应用统计学专业（本科）','专业： 电子科学与技术']
    tokenizes = get_tokenize(test_text)
    x = txt2feature_predict(word2vec, tokenizes)
    test_text = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=8, padding='post', dtype='float', truncating='post')
    test_text = np.array(test_text)
    model.save_weights(JOB_NAME_EXTRACTION_MODEL_PATH)
    model1, _ = create_model()
    model1.load_weights(JOB_NAME_EXTRACTION_MODEL_PATH)
    pre_y = model1.predict(test_text)
    print(pre_y)
    res = get_entity(pre_y, tokenizes)
    print(res)
    # print(scores)

    # freq_analyze(job_info)


def test2():
    file_path = ROOT_PATH + '\\job_order.txt'
    job_info = load_job(file_path)
    stop_words = stopwordslist()
    stop_words.append(' ')
    jieba.load_userdict(USER_DIC_PATH)
    for i in range(job_info.shape[0]):
        text = ' '.join(job_info.iloc[i, 1])
        text = re.sub(DEL_SPECIAL_PAT2, '', text)
        text_list = jieba.lcut(text)
        text_list = [text for text in text_list if text != ' ']
        job_info.iloc[i, 1] = text_list
    with open(ROOT_PATH + '\\job_corpus.txt', 'w', encoding='utf-8') as f:
        res = []
        for i in range(job_info.shape[0]):
            word_list = job_info.iloc[i, 1]
            res.append(' '.join(word_list))
        for r in res:
            print(r)
        f.writelines(res)


if __name__ == '__main__':
    test1()
    # corpus_processing1(ROOT_PATH + '\\专业大全.txt')
    # test_create_word2vec_corpus()








