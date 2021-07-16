
import numpy as np
from intro_block_classification.share_func import *
from constant.special_string import *
import jieba
import gensim
from sklearn.svm import SVC
import joblib
import pandas as pd
np.set_printoptions(suppress=True)

"""
项目经历需要提取的内容有：1.项目起始时间和结束时间 2.项目名称 3.担任角色 4.项目描述
如果有多段经历，输出一个列表
"""

# -----------------------------------------------------项目经历---------------------------------------- #


def create_data(file_path_pos, file_path_nag, to_path):
    with open(file_path_nag, 'r', encoding='utf-8') as f:
        data_list = f.readlines()
        data_list = [re.sub('\n', '', line) for line in data_list
                     if line != '********************************************************************\n']
        data_list = [line + '###0\n' for line in data_list]
        with open(to_path, 'a+', encoding='utf-8') as f1:
            f1.writelines(data_list)
    with open(file_path_pos, 'r', encoding='utf-8') as f:
        data_list = f.readlines()
        data_list = [re.sub('\n', '', line) for line in data_list if line != '\n']
        data_list = [line + '###1\n' for line in data_list]
        # print(data_list)
        with open(to_path, 'a+', encoding='utf-8') as f1:
            f1.writelines(data_list)


def load_data(file_path, seg='###'):
    """
    加载二分类模型数据集
    :param file_path:
    :return:train_data, train_label, test_data, test_label
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data_list = f.readlines()
        for i, line in enumerate(data_list):
            one_sample = line.split(seg)
            data_list[i] = {'text': one_sample[0], 'label': int(one_sample[1])}
    # print(data_list)
    random.shuffle(data_list)
    data_frame = pd.DataFrame(data=data_list, columns=['text', 'label'])
    train_len = data_frame.shape[0] * 3 // 5
    train_data, train_label, test_data, test_label = data_frame.iloc[:train_len, 0], data_frame.iloc[:train_len, 1], data_frame.iloc[train_len:, 0], data_frame.iloc[train_len:, 1]
    return train_data, train_label, test_data, test_label


def word2feature(text, word2vec_model=word2vec_model_path_2021_4_2):
    """
    将text转为向量
    :param text:str
    :param word2vec_model:word2vec模型
    :return:
    """
    stop_words = stopwordslist()
    jieba.load_userdict(USER_DIC_PATH)
    text = re.sub(DEL_SPECIAL_PAT, '', text)
    texts = jieba.lcut(text)
    texts = [t for t in texts if t not in stop_words]
    if isinstance(word2vec_model, str):
        word2vec_model = gensim.models.Word2Vec.load(word2vec_model_path_2021_4_2)
    vec = np.zeros(shape=(100, ))
    cnt = 0
    for t in texts:
        try:
            vec += np.array(word2vec_model[t])
            cnt += 1
        except KeyError:
            print('keyError:', t)
            continue
    if cnt == 0:
        return list(vec)
    return list(vec / cnt)


def word2feature_for_data_set(texts, word2vec_model=word2vec_model_path_2021_4_2):
    """

    :param texts: series
    :param word2vec_model: word2vec模型
    :return:向量list
    """
    if isinstance(word2vec_model, str):
        word2vec_model = gensim.models.Word2Vec.load(word2vec_model_path_2021_4_2)
    vecs = []
    for text in texts:
        vecs.append(word2feature(text, word2vec_model))

    return vecs


def model_train(file_path):
    model = SVC()
    X, y, test_x, test_y = load_data(file_path)  # series类型
    X = word2feature_for_data_set(X)
    y = list(y)
    test_x = word2feature_for_data_set(test_x)
    test_y = list(test_y)
    # print(len(test_y), len(test_x))
    model.fit(X, y)
    joblib.dump(model, ITEM_NAME_MODEL_PATH)
    model = joblib.load(ITEM_NAME_MODEL_PATH)
    predicts = model.predict(test_x)
    cnt = 0
    for i, pre in enumerate(predicts):
        # print(pre)
        if pre == test_y[i]:
            cnt += 1
    print('accuracy:', str(cnt / len(test_y)))


def split_text(text):
    """
    对于输入的一行文本，对其进行分割，返回最小实体组成的列表
    分割依据:
    1.如果文本长度在40以上则判定为项目描述，使用关键词“项目职责”、“项目描述”、“项目角色”、“项目业绩”、“项目成果”进行分割
    2.如果文本长度小于40且包含多个空格，则判定为项目头部信息，按空格分开
    :param text:str
    :return:
    """
    txt = re.sub(' ', '', text)
    if len(txt) <= 60:  # 项目头部
        if re.search('[ ]{3}', text) is not None:
            res = re.split(r"[ ]{3,20}", text)
            return res
        else:  # 说明只有一个实体， 如果文本长度大于30，则归类为项目具体内容
            if len(txt) <= 30:
                return [text]
    # 项目具体内容
    pattern = '项目职责[:：]*|项目描述[:：]*|项目角色[:：]*|项目业绩[:：]*|项目成果[:：]*'
    keys = re.search(pattern=pattern, string=text)
    if keys is not None:
        # print(keys)
        res = re.split(pattern='('+pattern+')', string=text)
        res1 = []
        tmp = ''
        i = 0
        if len(res) >= 1:
            if re.search(pattern, res[i]) is None:
                i += 1
        while i < len(res):
            tmp += res[i]
            i += 1
            if i < len(res):
                tmp += res[i]
            i += 1
            res1.append(tmp)
            tmp = ''
        return res1
    else:
        return [text]


def split_items(items):
    """
    将多个items中的每一个item分隔开
    分割依据:依次检索每一个行文本，检测项目头部
    项目头部特征：1.包含时间戳 2.短文本标题 3. 包含空格 4.不包含：项目描述、项目内容、项目介绍之类的关键词
    :param items: [row_text1, row_text2,..., row_textn]
    :return: [[row_text1, row_text2,...], [row_text1, row_text2,...], ...,[row_text1, row_text2,...]]
    """
    res = []
    one_item = []
    pattern = r'项目描述|项目内容|项目简介|项目介绍|项目概要|项目详情|项目概况|业绩|成果'
    is_head = False
    # 只有当从非项目头部状态转移至头部状态时，标志着一个项目结束
    for row in items:
        if detect_obtain_time([row]) is True or len(re.sub(' ', '', row)) <= 35 and re.search(pattern, row) is None:
            # 说明是项目头部信息
            if len(one_item) != 0 and not is_head:  # 上一个状态非头部
                res.append(one_item.copy())
                one_item.clear()
            is_head = True
            one_item.append(row)
        else:
            one_item.append(row)
            is_head = False
    if len(one_item) != 0:
        res.append(one_item)
    return res


def get_smallest_detect_units(item):
    """
    对于一个项目经历，得到最小检测单元，以列表形式返回
    :param item: [row_text1, row_text2, ..., row_textn]
    :return: [unit1, unit2, unit3,...unitn], unit:str
    """
    ans = []
    for row in item:
        tmp = split_text(row)
        ans.extend(tmp)
    return ans


def get_smallest_detect_units_for_all_items(items):
    """
    对于所有项目经历，得到最小检测单元，以列表形式返回
    :param items: [ [row1, row2, ..., rown], ... ,[row1, row2, ..., rown] ]
    :return: list(list(str)), [ [unit1, unit2, unit3,...unitn], ..., [unit1, unit2, unit3,...unitn] ], unit:str
    """
    ans = []
    for item in items:
        ans.append(get_smallest_detect_units(item))
    return ans


def get_smallest_detect_units_for_all_items_origin_rows(texts):
    """
    由原始文本（简历中每一行组成的行文本列表）作为输入，得到待检测的最小实体列表
    :param texts: [row1, row2, ..., rown]
    :return: [ [unit1, unit2, unit3,...unitn], ..., [unit1, unit2, unit3,...unitn] ], unit:str
    """
    items = split_items(texts)
    ans = []
    for item in items:
        ans.append(get_smallest_detect_units(item))
    return ans


def extract_item(texts, word2vec=word2vec_model_path_2021_4_2):
    """
    提取项目经历中的内容项：项目描述 项目名称、项目时间、项目角色
    :param texts:句子列表
    :param word2vec: 词向量模型
    :return:[{start_time:xx, end_time:xx, item_name:xx, item_role:xx, item_desc:xx}, {}, ..., {}]
    """
    ans = []
    item_name_model = joblib.load(ITEM_NAME_MODEL_PATH)
    if isinstance(word2vec, str):
        word2vec = gensim.models.Word2Vec.load(word2vec)
    items = get_smallest_detect_units_for_all_items_origin_rows(texts)  # 得到所有项目实体
    for item in items:
        # print(item)
        dic = {'start_time': '', 'end_time': '', 'item_name': '', 'item_desc': '', 'item_role': ''}
        detect_obtain_time(item, dic, delete=True)
        obtain_item_name(item, item_name_model, dic, delete=True, word2vec=word2vec)
        obtain_item_role(item, dic, delete=True)
        dic['item_desc'] = '\n'.join(item)
        ans.append(dic)
    return ans


def obtain_item_name(texts, model, dic=None, delete=False, word2vec=word2vec_model_path_2021_4_2):
    """
    获取项目名称，利用二分类模型判断
    :param texts: [unit1, unit2, ..., unitn]
    :param dic: {'item_name':xx}
    :param delete: 默认为False,表示不对texts进行修改
    :param model:二分类模型
    :param word2vec 词向量模型
    :return: None
    """
    if isinstance(word2vec, str):
        word2vec = gensim.models.Word2Vec.load(word2vec)
    for i, unit in enumerate(texts):
        if model.predict([word2feature(unit, word2vec)])[0] == 1:
            dic['item_name'] = re.sub(r'项目名称[:： \-]*|名称[:： \-]*|项目名[:： \-]*|[ ]*', '', unit)
            if delete:
                texts.pop(i)
            break


def obtain_item_role(texts, dic=None, delete=True):
    """
    获取项目角色，利用规则提取
    :param texts: [unit1, unit2, ..., unitn]
    :param dic: {'item_role':xx}
    :param delete: 默认为False,表示不对texts进行修改
    :return: None
    """
    pattern = '项目角色[:：]*|担任角色[:：]*|担任职位[:：]*|担任职责[:：]*|项目职责[:：]*|角色[:：]*|职位[:：]*|职责[:：]*'
    for i, unit in enumerate(texts):
        if len(unit) <= 30 and re.search(pattern, unit) is not None:
            dic['item_role'] = re.sub(pattern, '', unit)
            if delete:
                texts.pop(i)
            break


def detect_obtain_time(texts, dic=None, delete=False):
    """
    通过规则提取时间, dic[start_time] = xx, dic[end_time] = xx
    :param texts:[unit1, unit2, unit3,...,unitm]
    :param dic:{ }
    :param delete:默认不对texts修改,True表示一旦检测到时间戳对应行，将该行从texts中删除
    :return:None
    """
    pat1 = r"(\d{4}[ ]{0,3}/[ ]{0,3}\d{1,2}[ ]{0,3}"\
                        r"/?[ ]{0,3}\d{0,2}[ ]{0,3})[ ]{0,3}[-到至]{0,3}[ ]{0,3}(\d{4}[ ]{0,3}"\
                        r"/[ ]{0,5}\d{1,2}[ ]{0,3}/?[ ]{0,3}\d{0,2}[ ]{0,3})?"
    pat2 = r"(\d{4}[ ]{0,3}\.[ ]{0,3}\d{1,2}[ ]{0,3}"\
                        r"\.?[ ]{0,3}\d{0,2}[ ]{0,3})[ ]{0,3}[-到至]{0,3}[ ]{0,3}(\d{4}[ ]{0,3}"\
                        r"\.[ ]{0,5}\d{1,2}[ ]{0,3}\.?[ ]{0,3}\d{0,2}[ ]{0,3})?"
    pat3 = r"(\d{4}[ ]{0,3}-[ ]{0,3}\d{1,2}[ ]{0,3}"\
                        r"-?[ ]{0,3}\d{0,2}[ ]{0,3})[ ]{0,3}[-到至]{0,3}[ ]{0,3}(\d{4}[ ]{0,3}"\
                        r"-[ ]{0,5}\d{1,2}[ ]{0,3}-?[ ]{0,3}\d{0,2}[ ]{0,3})?"
    res = None
    for i, text in enumerate(texts):
        res = re.search(pat1, text)
        if res is None:
            res = re.search(pat2, text)
            if res is None:
                res = re.search(pat3, text)
        if dic is not None and res is not None:
            start = res.group(1)
            end = res.group(2)
            if start is not None:
                dic['start_time'] = start
            else:
                dic['start_time'] = ' '
            if end is not None:
                dic['end_time'] = end
            else:
                dic['end_time'] = ' '
            if delete:
                texts[i] = re.sub(pat1, '', text)
                texts[i] = re.sub(pat2, '', text)
                texts[i] = re.sub(pat3, '', text)
                if len(texts[i]) < 3:
                    texts.pop(i)
            break
    if res is not None:
        return True
    return False


def test_split_text():
    text = """项目描述:该项目为南京航空航天大学实验室科研实践项目，主要工作为对 Eclipse 中的用例文档添加领域专业名词智能提
        示功能，改善用户个性化体验，涉及文本聚类、文本分类、关键词抽取、序列标注任务。核心技术为 KMeans、KNN、CRF。
        项目成果：提出增量式文本聚类算法以及用 KNN 进行无监督式领域分类，提出 tf-idf 过滤算法以提高关键词抽取准确率
        提出词性预测模型以提高专业名词提示结果的 f1-score 值，最终 f1-score 比已有方法高出 3%左右，另有一篇论文在投。项目职责:"""
    text1 = '2017.02.1 - 2019.03.23       南京航空航天大学实验室科研项目   担任职责：核心算法设计与开发'
    res = split_text(text1)
    print(res)


def test_split_items():
    items = [
        """2017.02.1 - 2019.03.23""",
        """南京航空航天大学实验室科研项目""",
        """担任职责：核心算法设计与开发""",
        """项目描述:该项目为南京航空航天大学实验室科研实践项目，主要工作为对 Eclipse 中的用例文档添加领域专业名词智能提
        示功能，改善用户个性化体验，涉及文本聚类、文本分类、关键词抽取、序列标注任务。核心技术为 KMeans、KNN、CRF。
        项目成果：提出增量式文本聚类算法以及用 KNN 进行无监督式领域分类，提出 tf-idf 过滤算法以提高关键词抽取准确率
        提出词性预测模型以提高专业名词提示结果的 f1-score 值，最终 f1-score 比已有方法高出 3%左右，另有一篇论文在投。项目职责:""",
        """简历文档信息抽取   担任职责：架构设计""",
        """2017.02.1 - 至今""",
        """项目描述:该项目为南京航空航天大学实验室科研实践项目，主要工作为对 Eclipse 中的用例文档添加领域专业名词智能提
                示功能，改善用户个性化体验，涉及文本聚类、文本分类、关键词抽取、序列标注任务。核心技术为 KMeans、KNN、CRF。
                项目成果：提出增量式文本聚类算法以及用 KNN 进行无监督式领域分类，提出 tf-idf 过滤算法以提高关键词抽取准确率"""

        ]
    res = split_items(items)
    for item in res:
        print(item)


def test_extract_items():
    texts = [
        """2017.02.1 - 2019.03.23""",
        """南京航空航天大学实验室科研项目""",
        """担任职责：核心算法设计与开发""",
        """项目描述:该项目为南京航空航天大学实验室科研实践项目，主要工作为对 Eclipse 中的用例文档添加领域专业名词智能提
        示功能，改善用户个性化体验，涉及文本聚类、文本分类、关键词抽取、序列标注任务。核心技术为 KMeans、KNN、CRF。
        项目成果：提出增量式文本聚类算法以及用 KNN 进行无监督式领域分类，提出 tf-idf 过滤算法以提高关键词抽取准确率
        提出词性预测模型以提高专业名词提示结果的 f1-score 值，最终 f1-score 比已有方法高出 3%左右，另有一篇论文在投。项目职责:""",
        """简历文档信息抽取   担任职责：架构设计""",
        """2017.02.1 - 至今""",
        """项目描述:该项目为南京航空航天大学实验室科研实践项目，主要工作为对 Eclipse 中的用例文档添加领域专业名词智能提
                示功能，改善用户个性化体验，涉及文本聚类、文本分类、关键词抽取、序列标注任务。核心技术为 KMeans、KNN、CRF。
                项目成果：提出增量式文本聚类算法以及用 KNN 进行无监督式领域分类，提出 tf-idf 过滤算法以提高关键词抽取准确率"""
    ]
    word2vec = gensim.models.Word2Vec.load(word2vec_model_path_2021_4_2)
    ans = extract_item(texts, word2vec)
    for item in ans:
        print('one item:')
        print(item)


if __name__ == '__main__':
        # test_split_text()
        # test_split_items()
        test_extract_items()
        # create_data(ROOT_PATH+'\\item_name.txt', file_path_nag=ROOT_PATH + '\\item_name_noise.txt',
        #             to_path=ROOT_PATH + '\\item_name_data_set.txt')
        # X, y, t_x, t_y = load_data(ITEM_NAME_DATA_SET_PATH)
        # print(X.shape)
        # print(t_x.shape)
        # model_train(ITEM_NAME_DATA_SET_PATH)


