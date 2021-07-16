
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from constant.special_string import *
import joblib
from intro_block_classification.crf_torch import CRF
from torch.utils.data import random_split
import re
from torch.nn.init import zeros_
import numpy as np
import torch.nn.functional as f
import random
from pytorch_learning.transformer import Encoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SkillExtractionModelTransformerCRF(nn.Module):
    def __init__(self, embedding_dim=128, vocab_size=4000, block_num=1, key_dim=128,
                 value_dim=128, dim_model=128, n_class=5, head_num=8, n_position=150):
        super(SkillExtractionModelTransformerCRF, self).__init__()
        self.crf = CRF(n_class=n_class)
        self.loss_fun = self.crf.neg_log_likelihood_loss
        # 8 params
        self.transformer_encoder = Encoder(key_dim=key_dim, v_dim=value_dim, vocab_size=vocab_size,
                                           block_num=block_num, dim_model=dim_model,
                                           embed_dim=embedding_dim, head_num=head_num, n_position=n_position)
        # 定义模型结构： 字符级embedding:transormer + crf
        self.linear = nn.Linear(in_features=dim_model, out_features=n_class)
        self.layer_norm = nn.LayerNorm([n_class])

    def forward(self, x, mask=None):
        """
        :param x: (batch_size, time_step) ,元素为字符索引
        :param mask:掩码，提供真实序列长度, (batch_size, time_step)
        :return: prey
        """
        # print('original mask.shape:', mask.shape)
        # print(mask)
        out = self.transformer_encoder(x, mask=mask)
        out = self.linear(out)  # (batch_size, time_step, n_class)
        # print('mask.shape:', mask.shape)
        # print(mask)
        out = self.layer_norm(out)
        res = self.crf(out, mask)
        # print(out)
        return out, res  # 返回每一个样本的标签序列[ (score, [label1, label2,...]) , ... ]


class SkillExtractionModel(nn.Module):
    def __init__(self, embedding_dim=100, vocab_size=5000, num_layers=1, hidden_size=64, n_class=5):
        super(SkillExtractionModel, self).__init__()
        self.crf = CRF(n_class=n_class)
        self.loss_fun = self.crf.neg_log_likelihood_loss
        # 定义模型结构： 字符级embedding + 1层双向lstm + crf
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, num_layers=num_layers, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=n_class)
        self.layer_normal = nn.LayerNorm([hidden_size * 2])

    def init_param(self):
        """
        初始化权重
        :return:
        """
        # 初始化embedding层
        initrange = np.sqrt(3.0 / self.embedding_dim)
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # 初始化线性层
        initrange = np.sqrt(6.0 / (self.linear.weight.size(0) + self.linear.weight.size(1)))
        self.linear.weight.data.uniform_(-initrange, initrange)
        zeros_(self.linear.bias.data)

    def forward(self, x, mask=None):
        """
        :param x: (batch_size, time_step) ,元素为字符索引
        :param mask:掩码，提供真实序列长度
        :return: prey
        """
        # print(x.device)
        # x = x.to('cpu')
        embed = self.embedding(x)   # (batch_size, time_step, embed_dim) nn.Embedding 会自动增加一维
        # embed = embed.to(device)
        out, _ = self.lstm(embed)  # (batch_size, time_step, hidden_size*2)
        out = self.layer_normal(out)
        out = self.linear(out)  # (batch_size, time_step, n_class)
        res = self.crf(out, mask)
        return out, res  # 返回每一个样本的标签序列[ (score, [label1, label2,...]) , ... ]


class SkillData(Dataset):
    def __init__(self, x=None, y=None, file_path=None, ch_2_num=None, num_2_ch=None):
        self.x = []  # list(list(int))
        self.y = []  # list(list(int))
        self.ch_2_num = {} if ch_2_num is None else ch_2_num
        self.label_2_num = {'B': 0, 'I': 1, 'O': 2, 'S': 3, 'T': 4}
        self.num_2_label = ['B', 'I', 'O', 'S', 'T']  # 开始、内部、其他、独立、提示词
        self.num_2_ch = ['START'] if num_2_ch is None else num_2_ch
        self.vocab_size = len(self.num_2_ch)  # 默认有一个特殊标记符，对应padding_idx=0
        # 行分割符号为:###
        count = 0
        assert x is not None and y is not None or file_path is not None
        if x is not None and y is not None:
            self.x, self.y = [], []
            for i, row_list in enumerate(x):
                tmp_x, tmp_y = [], []
                for j, ch in enumerate(row_list):
                    if ch not in self.ch_2_num:
                        self.ch_2_num[ch] = self.vocab_size
                        self.vocab_size += 1
                        self.num_2_ch.append(ch)
                    tmp_x.append(self.ch_2_num[ch])
                    tmp_y.append(self.label_2_num[y[i][j]])
                self.x.append(tmp_x)
                self.y.append(tmp_y)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                tmp_x, tmp_y = [], []
                for line in f:
                    line = re.sub('[ ]+', ' ', line).strip(' ')
                    line = re.sub(r'[ ]+\n', '', line)
                    line = line.strip('\n')
                    if line == '###':
                        self.x.append(tmp_x)
                        self.y.append(tmp_y)
                        tmp_x, tmp_y = [], []
                    else:
                        # print('count ', count,  ':', repr(line))
                        i = len(line) - 1
                        while i >= 0 and line[i] not in self.label_2_num:
                            i -= 1
                        assert i >= 0
                        ch, label = line[:i].strip(' '), line[i]
                        if ch not in self.ch_2_num:
                            self.ch_2_num[ch] = self.vocab_size
                            self.vocab_size += 1
                            self.num_2_ch.append(ch)
                        tmp_x.append(self.ch_2_num[ch])
                        tmp_y.append(self.label_2_num[label])
                    count += 1
                if len(tmp_x) != 0:
                    self.x.append(tmp_x)
                    self.y.append(tmp_y)
        print('x.length:', len(self.x), 'y.length:', len(self.y))

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        return x, y

    def __len__(self):
        return len(self.x)

    def update(self, x, y):
        """

        :param x: list(list(str))
        :param y: list(list(str))
        :return:None
        """
        for i, row_list in enumerate(x):
            tmp_x, tmp_y = [], []
            for j, ch in enumerate(row_list):
                if ch not in self.ch_2_num:
                    self.ch_2_num[ch] = self.vocab_size
                    self.vocab_size += 1
                    self.num_2_ch.append(ch)
                tmp_x.append(self.ch_2_num[ch])
                tmp_y.append(self.label_2_num[y[i][j]])
            self.x.append(tmp_x)
            self.y.append(tmp_y)


def load_data(file_path):
    skill_data = SkillData(file_path)
    data = skill_data[1]
    print(data)


def compute_loss_weight(loaders, num_2_label, label_2_num):
    count = {'B': 0, 'I': 0, 'O': 0, 'S': 0, 'T': 0}
    for loader in loaders:
        for _, y, mask in loader:  # 1 batch
            # y.shape:(batch_size, max_len)
            seq_len = torch.sum(mask, dim=1)  # seq_len
            pack_y = pack_padded_sequence(y, lengths=seq_len.cpu(), batch_first=True)  # 一维 (batch_size * seq_len,)
            for num in pack_y.data:
                try:
                    count[num_2_label[num.item()]] += 1
                except KeyError:
                    count[num_2_label[num.item()]] = 0
    sum_cnt = sum(count.values())
    weights = [0 for _ in range(len(num_2_label))]
    for key in count:
        weights[label_2_num[key]] = (sum_cnt - count[key])/sum_cnt
    return count, weights


def train_model(model, data, optimizer, mode='lstm'):
    """
    一个epoch 的操作：前向计算(forward)， 梯度计算(loss.backward)，梯度更新(optimizer.step)
    :param data:
    :param model:
    :param optimizer
    :param mode:lstm or crf
    :return:
    """
    total_loss = 0
    count = 0
    lengths = 0
    model.train()  # 调为训练模式
    for i, (x, y, mask) in enumerate(data):  # 迭代data(dataLoader对象)中的每一个batch
        # torch.autograd.set_detect_anomaly(True)
        out_lstm, out_crf = model(x, mask)  # (batch_size, time_step, n_class),
        # list(tuple(float, list(str))) len(list) = batch_size
        # y.shape:(batch_size, time_step)
        if mode == 'crf':
            seq_lens = torch.sum(mask, dim=1)
            batch_loss = model.loss_fun(x=out_lstm, mask=mask, target=y)
            incre_count, incre_length = compute_acc_crf(out_crf, y, seq_lens)
        else:
            seq_lens = torch.sum(mask, dim=0)
            incre_count, incre_length, batch_loss = compute_acc_loss_lstm(out_lstm, y, seq_lens)
        total_loss += batch_loss
        count += incre_count
        lengths += incre_length
        # with torch.autograd.detect_anomaly():
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)  # 梯度裁剪
        optimizer.step()
        # print(i+1)
    return {'train_accuracy': count / lengths, 'train_loss': total_loss}


def compute_acc_crf(out_crf, y, seq_lens):
    count, lengths = 0, 0
    with torch.no_grad():
        # 统计准确率
        for i, (_, path, _) in enumerate(out_crf):  # 迭代一个batch
            path = torch.tensor(path, dtype=torch.int64).cuda()
            target = y[i, :seq_lens[i].item()]
            count += torch.sum(path == target).item()
            lengths += target.shape[0]
    return count, lengths


def compute_acc_loss_lstm(out_lstm, y, seq_lens, loss_compute=True):
    """

    :param out_lstm: shape:(batch_size, time_step, n_class)
    :param y: (batch_size, time_step)
    :param seq_lens: (batch_size)
    :param loss_compute : 是否计算loss， 默认True
    :return: loss, count, lengths
    """
    # (batch_size * time_step, n_class)
    out_lstm = torch.nn.utils.rnn.pack_padded_sequence(out_lstm, lengths=seq_lens.cpu(), batch_first=False)
    pack_y = torch.nn.utils.rnn.pack_padded_sequence(y, lengths=seq_lens.cpu(), batch_first=False)  # (batch_size * time_step)
    loss = 0
    if loss_compute:
        loss = f.cross_entropy(out_lstm.data, pack_y.data)
    with torch.no_grad():
        out_lstm = torch.argmax(out_lstm.data, dim=1)
        count = torch.sum(out_lstm.data == pack_y.data).item()
        lengths = pack_y.data.shape[0]
    if loss_compute:
        return count, lengths, loss
    else:
        return count, lengths


def evaluation(model, val_loader, mode='lstm'):
    model.eval()  # 等价于model.train(False)
    total_loss = 0
    count = 0
    lengths = 0
    with torch.no_grad():
        for idx, (x, y, mask) in enumerate(val_loader):
            out_lstm, out_crf = model(x, mask)  # (batch_size, time_step, n_class),
            # list(tuple(float, list(str))) len(list) = batch_size
            # y.shape:(batch_size, time_step)
            if mode == 'crf':
                seq_lens = torch.sum(mask, dim=1)
                loss = model.loss_fun(x=out_lstm, mask=mask, target=y)
                incre_count, incre_length = compute_acc_crf(out_crf, y, seq_lens)
            else:
                seq_lens = torch.sum(mask, dim=0)
                incre_count, incre_length, loss = compute_acc_loss_lstm(out_lstm, y, seq_lens)
            lengths += incre_length
            count += incre_count
            total_loss += loss
        epoch_acc = count / lengths
    return {'valid_accuracy': epoch_acc, 'valid_loss': total_loss}


def test_model(model, data):
    pass


def collate_fn(batch):
    """
    将不定长序列处理为定长序列
    :param batch: batch list. list( tuple( list(int), list(int) ) )
    :return:
    """
    processed = []
    for (x, y) in batch:
        # x:list(int)
        processed.append([torch.tensor(x), torch.tensor(y)])
    processed.sort(key=lambda k: k[0].shape[0], reverse=True)
    inputs = [x[0] for x in processed]
    labels = [x[1] for x in processed]
    max_len = inputs[0].shape[0]
    mask = torch.ones((len(inputs), max_len), dtype=torch.int64)  # (batch_size, max_len)
    for i in range(mask.shape[0]):
        mask[i, inputs[i].shape[0]:max_len] -= 1
    pad_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    pad_labels = pad_sequence(labels, batch_first=True, padding_value=3)
    # pad_inputs.shape:(batch_size, max_len)
    # pad_labels.shape:(batch_size, max_len)
    return pad_inputs.to(device), pad_labels.to(device), mask.to(device)


def data_augumentation(data):
    """
    数据增强，在原有的样本基础上，选择40%的样本数量以一定的概率将字符替换成MASK_CH/MASK_ENG
    :param data:SKillData
    :return:
    """
    tmp_x, tmp_y = [], []
    for i, (x, y) in enumerate(data):
        p = random.random()
        if p > 0.7:
            # 0.4的概率产生新样本
            num = int(random.uniform(0.1, 0.2) * len(x))  # 改为Mask的数量
            indexes = random.sample([i for i in range(len(x))], num)
            x = [data.num_2_ch[x[i]] for i in range(len(x))]  # 还原为字符
            for index in indexes:  # MASK填充
                if re.search(pattern=r'[a-zA-Z]+', string=x[index]) is None:
                    x[index] = SP_CH
                else:
                    x[index] = SP_ENG
            tmp_x.append(x)
            tmp_y.append([data.num_2_label[y[i]] for i in range(len(y))])
    data.update(tmp_x, tmp_y)


def add_mask(text, ch_2_num=None):
    """
    将未出现的字符用MASK_CH/MASK_ENG表示
    :param text: str
    :param ch_2_num:
    :return:list(str) the tokens of text
    """
    if ch_2_num is None:
        ch_2_num = joblib.load(VOCAB_PATH)['ch_2_num']
    pat = r'[a-zA-Z][a-zA-Z.0-9]*'
    ch_arr = []
    res = re.finditer(string=text, pattern=pat, flags=re.IGNORECASE)
    i = 0
    for item in res:
        if i < item.start():
            for j in range(i, item.start()):
                if text[j] in ch_2_num:
                    ch_arr.append(text[j])
                else:
                    ch_arr.append(SP_CH)
            # print(item.group())
        if item.group() in ch_2_num:
            ch_arr.append(item.group())
        else:
            ch_arr.append(SP_ENG)
        i = item.end()
    for j in range(i, len(text)):
        if text[j] in ch_2_num:
            ch_arr.append(text[j])
        else:
            ch_arr.append(SP_CH)
    return ch_arr


def trans_x(input, ch_2_num):
    """
    预测时，将输入数据转为符合模型输入格式的数据
    当遇到标识符（英文数字下划线组合），将标识符作为一个整体
    :param input: list(str)
    :param ch_2_num: 字符转为索引
    :return:
    """
    for i, text in enumerate(input):
        input[i] = add_mask(text)
    pad_val = 0
    for i, sen in enumerate(input):
        tmp = []
        for ch in sen:
            tmp.append(ch_2_num[ch])
        input[i] = tmp
    seq_lens = [len(sen) for sen in input]
    max_len = max(seq_lens)
    for i in range(len(input)):
        for _ in range(max_len-len(input[i])):
            input[i].append(pad_val)
    x = torch.tensor(input)
    mask = torch.zeros([x.shape[0], max_len], dtype=torch.int64)
    for i in range(mask.shape[0]):
        mask[i, :seq_lens[i]] += 1
    return x, mask


if __name__ == '__main__':
    # load_data(file_path)

    def train_process():
        choose = int(input('chose model type:1-lstm + crf, 2-transformer+crf\n'))
        vocab = joblib.load(VOCAB_PATH)
        train = SkillData(file_path=SKILL_DATA_TRAIN, num_2_ch=vocab['num_2_ch'], ch_2_num=vocab['ch_2_num'])
        test = SkillData(file_path=SKILL_DATA_TEST, num_2_ch=vocab['num_2_ch'], ch_2_num=vocab['ch_2_num'])
        valid = SkillData(file_path=SKILL_DATA_VALID, num_2_ch=vocab['num_2_ch'], ch_2_num=vocab['ch_2_num'])
        # joblib.dump({'ch_2_num': valid.ch_2_num, 'num_2_ch': valid.num_2_ch}, VOCAB_PATH)
        batch_size = 32
        train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        valid_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        epoch_num = 20
        batch_len_train, batch_len_val, batch_len_test = len(train) // batch_size,\
                                                         len(valid) // batch_size, len(test) // batch_size
        # print(batch_len_train)
        if choose == 1:
            model = SkillExtractionModel(vocab_size=train.vocab_size).cuda()
        else:
            model = SkillExtractionModelTransformerCRF(vocab_size=train.vocab_size).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(epoch_num):
            train_res = train_model(model, train_loader, optimizer, mode='crf')
            eval_res = evaluation(model, valid_loader, mode='crf')
            print(f"eopch{ epoch}, train accuracy:{train_res['train_accuracy']}, "
                  f"train loss:{train_res['train_loss'] / batch_len_train}, "
                  f"valid accuracy:{eval_res['valid_accuracy']}, "
                  f"valid loss:{eval_res['valid_loss']}")
        # 保存
        if choose == 1:
            torch.save(model.state_dict(), ROOT_PATH + '\\extract_skill_model_20210607.pkl')
        else:
            torch.save(model.state_dict(), ROOT_PATH + '\\extract_skill_model_transformer_crf_20210607.pkl')

    def test_forward():
        model = SkillExtractionModel(vocab_size=100)
        model.to('cuda')
        input = torch.tensor([[1, 2, 3], [4, 5, 0], [6, 0, 0]], dtype=torch.int64).cuda()
        mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=torch.int64).cuda()
        out_lstm, out_crf = model(input, mask)
        print(out_lstm)
        print(out_crf)

    def process_train_txt():
        txt = ''
        with open(SKILL_DATA_TRAIN, 'r', encoding='utf-8') as f:
            txt = f.read()
        txt = txt.split('###')
        print(len(txt))
        print(txt)

    def split_data():
        """
        train: val:test = 6:1:3
        :return:
        """
        data = SkillData(file_path=SKILL_DATA_TRAIN)
        data_augumentation(data)
        joblib.dump({'ch_2_num': data.ch_2_num, 'num_2_ch': data.num_2_ch}, VOCAB_PATH)
        train_len = int(len(data) * 0.7)
        val_len = int(len(data) * 0.1)
        test_len = len(data) - train_len - val_len
        train, validation, test = random_split(data, lengths=[train_len, val_len, test_len])
        with open(SKILL_DATA_TRAIN, 'w', encoding='utf-8') as f:
            for (x, y) in train:
                for i, elem in enumerate(x):
                    f.write(data.num_2_ch[elem] + ' ' + data.num_2_label[y[i]] + '\n')
                f.write('###\n')
        with open(SKILL_DATA_VALID, 'w', encoding='utf-8') as f:
            for (x, y) in validation:
                for i, elem in enumerate(x):
                    f.write(data.num_2_ch[elem] + ' ' + data.num_2_label[y[i]] + '\n')
                f.write('###\n')
        with open(SKILL_DATA_TEST, 'w', encoding='utf-8') as f:
            for (x, y) in test:
                for i, elem in enumerate(x):
                    f.write(data.num_2_ch[elem] + ' ' + data.num_2_label[y[i]] + '\n')
                f.write('###\n')

    def test_label_balance():
        """
        查看各个标签的比例
        :return:
        """
        vocab = joblib.load(VOCAB_PATH)
        train = SkillData(SKILL_DATA_TRAIN, num_2_ch=vocab['num_2_ch'], ch_2_num=vocab['ch_2_num'])
        test = SkillData(SKILL_DATA_TEST, num_2_ch=vocab['num_2_ch'], ch_2_num=vocab['ch_2_num'])
        valid = SkillData(SKILL_DATA_VALID, num_2_ch=vocab['num_2_ch'], ch_2_num=vocab['ch_2_num'])
        batch_size = 64
        train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        valid_loader = DataLoader(dataset=valid, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        count, weights = compute_loss_weight([train_loader, test_loader, valid_loader], num_2_label=train.num_2_label,
                                             label_2_num=train.label_2_num)
        print(count)
        print(weights)
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = torch.softmax(weights, dim=0)
        print('after softmax:', weights.numpy())

    def test_model_predict():
        sentence = '熟练掌握大数据以及数据铝构等专业节能'
        # 加载
        vocab = joblib.load(VOCAB_PATH)
        num_2_ch, ch_2_num = vocab['num_2_ch'], vocab['ch_2_num']
        x, mask = trans_x([sentence], ch_2_num=ch_2_num)
        x, mask = x.cuda(), mask.cuda()
        model = SkillExtractionModel(vocab_size=len(vocab['num_2_ch'])).cuda()
        model.load_state_dict(torch.load(EXTRACT_SKILL_MODEL_PATH))
        model.eval()
        _, res = model(x, mask)
        print(res)


    def test_split_setence():
        ch_2_num = joblib.load(VOCAB_PATH)['ch_2_num']
        sentence = '精通孙子兵法'
        pat = r'[a-zA-Z][a-zA-Z.0-9]*'
        ch_arr = []
        res = re.finditer(string=sentence, pattern=pat, flags=re.IGNORECASE)
        i = 0
        for item in res:
            if i < item.start():
                for j in range(i, item.start()):
                    if sentence[j] in ch_2_num:
                        ch_arr.append(sentence[j])
                    else:
                        ch_arr.append(SP_CH)
                # print(item.group())
            if item.group() in ch_2_num:
                ch_arr.append(item.group())
            else:
                ch_arr.append(SP_ENG)
            i = item.end()
        for j in range(i, len(sentence)):
            if sentence[j] in ch_2_num:
                ch_arr.append(sentence[j])
            else:
                ch_arr.append(SP_CH)
        print(ch_arr)


    # test_forward()
    train_process()
    # process_train_txt()
    # split_data()
    # test_label_balance()
    # test_model_predict()
    # test_split_setence()

