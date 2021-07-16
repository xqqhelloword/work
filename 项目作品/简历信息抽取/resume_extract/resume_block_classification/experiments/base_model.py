import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from functools import wraps
from resume_block_classification.experiments.data_set.block_data import BlockData
from resume_block_classification.data_precess.create_data import number2label, label_number


def decorator(func):
    @wraps(func)
    def my_acc(mask, target, pred):
        # print('before get acc')
        return func(mask, target=target, pred=pred)

    return my_acc


class BaseModel(nn.Module):
    def __init__(self, time_step, feature_size, batch_size=5, epochs=10):
        super(BaseModel, self).__init__()
        self.time_step = time_step
        self.feature_size = feature_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.label_distribution: dict = {}
        self.weights = None

    def forward(self, x):
        pass

    def cct(self, actual: torch.tensor, pred: torch.tensor):
        """
        计算N个样本的平均交叉熵
        :param actual: (batch_size)
        :param pred: (batch_size, one_hot_size)
        :return:loss
        """
        if self.weights is not None:
            cross_entropy = F.cross_entropy(input=pred, target=actual, weight=self.weights)
        else:
            cross_entropy = F.cross_entropy(input=pred, target=actual)
        return cross_entropy

    def my_loss(self, actual: torch.Tensor, predict, mask):
        """
        交叉熵损失
        :param actual: (batch_size, time_step)
        :param predict: (batch_size, time_step, one_hot_size)
        :param mask: (batch_size, time_step)
        :return:
        """
        seq_lens = torch.sum(mask, dim=1)  # 得到每个样本的序列长度
        # actual = actual.view(actual.shape[0] * actual.shape[1])
        # predict = predict.view([predict.shape[0] * predict.shape[1], predict.shape[-1]])
        loss = torch.tensor(0.0, dtype=torch.float64).cuda()
        for i in range(actual.shape[0]):
            loss += self.cct(actual=actual[i, :seq_lens[i]], pred=predict[i, :seq_lens[i], :])
        return loss

    def train_step(self, inputs: DataLoader, optimizer, epoch_num, metric=None):  # 执行一个epoch
        """

        :param inputs: DataLoader
        :return:
        """

        if metric is None:
            metric = BaseModel.get_acc
        acc_count, total, total_loss = 0, 0, 0
        for i, (x, target, mask) in enumerate(inputs):
            # x.shape: (batch_size, time_step, feature_dim) target.shape:(batch_size, time_step)
            # mask.shape:(batch_size, time_step)
            pred = self.forward(x)
            loss = self.my_loss(target, pred, mask)
            loss.backward()
            with torch.no_grad():
                # print('before:', pred.shape)
                one_batch_acc_cnt, one_batch_total_cnt = metric(mask=mask, target=target, pred=pred)
                acc_count += one_batch_acc_cnt
                total += one_batch_total_cnt
                total_loss += loss.item()
            optimizer.step()
            # 计算metric
            template = 'Batch {}, Loss: {}, accuracy:{}%'
            print(template.format(i + 1, loss, acc_count / total))
        print('')
        template = 'Epoch {}, Loss:{}, accuracy:{}%'
        print(template.format(epoch_num, total_loss / len(inputs), acc_count / total))
        for i in range(10):
            print('')

    @staticmethod
    @decorator
    def get_acc(mask, target, pred):
        # y.shape:(batch_size * time_step)
        # acc_count:一个batch下正确的个数， total_count:样本的总个数
        acc_count, total_count = 0, 0
        # print(pred.shape)
        seq_lens = torch.sum(mask, dim=1)
        for j, seq_len in enumerate(seq_lens):
            y, y_pre = target[j, : seq_len], pred[j, :seq_len, :]
            acc_count += torch.sum(y == torch.argmax(y_pre, dim=-1), dim=-1).item()
            total_count += seq_len
        return acc_count, total_count

    def validate_step(self, inputs: DataLoader, epoch_num=1, metric=None, test_data=False):  # 执行一个epoch
        """

        :param inputs: DataLoader
        :param epoch_num: 第几个epoch
        :param test_data:是否为训练集/测试集
        :param metric: 评估指标函数
        :return:
        """
        if test_data:
            pat = 'test'
        else:
            pat = 'val'
        if not metric:
            metric = BaseModel.get_acc
        acc_count, total, total_loss = 0, 0, 0
        for i, (x, target, mask) in enumerate(inputs):
            # x.shape: (batch_size, time_step, feature_dim) target.shape:(batch_size, time_step)
            # mask.shape:(batch_size, time_step)
            pred = self.forward(x)
            loss = self.my_loss(target, pred, mask)
            total_loss += loss.item()
            # y.shape:(batch_size * time_step)
            batch_acc_count, batch_total_count = metric(mask=mask, target=target, pred=pred)
            acc_count += batch_acc_count
            total += batch_total_count

            # 计算metric

            template = 'Batch {}, {} Loss: {}, {} accuracy:{}%'
            print(template.format(i + 1, pat, loss, pat, acc_count / total))
        print('')
        template = 'Epoch {}, {} Loss:{}, {} accuracy:{}%'
        print(template.format(epoch_num, pat, total_loss / len(inputs), pat, acc_count / total))
        for i in range(10):
            print('')

    def fit(self, x_train, y_train, optimizer, x_val=None, y_val=None):
        self.train(True)
        data_set = BlockData(x=x_train, y=y_train, feature_size=self.feature_size, time_step=self.time_step,
                  type='rnn')
        self.label_distribution = data_set.label_distribution
        print('类别数量分布：', self.label_distribution)
        total = 0
        tmp = [0 for _ in range(len(number2label))]
        for key in self.label_distribution:
            total += self.label_distribution[key]
            tmp[label_number[key]] = self.label_distribution[key]
        for i, item in enumerate(tmp):
            tmp[i] = total - item
        mean_val = sum(tmp) / len(tmp)
        varriant = math.sqrt(sum([(item - mean_val) ** 2 for item in tmp]))
        self.weights = F.softmax(torch.tensor([(item - mean_val) / (varriant + 0.001) for item in tmp],
                                              dtype=torch.float32, requires_grad=False).cuda().detach())
        print('类别损失权重:', self.weights)

        train_ds = DataLoader(data_set, shuffle=True, collate_fn=BaseModel.collate_fn,
                              batch_size=self.batch_size)

        val_ds = None
        if x_val is not None and y_val is not None:
            val_ds = DataLoader(
                BlockData(x=x_val, y=x_val, feature_size=self.feature_size, time_step=self.time_step,
                          type='rnn'), shuffle=True, collate_fn=BaseModel.collate_fn)

        for epoch in range(self.epochs):
            print('--------------------Epoch', epoch+1, '/', self.epochs, '-----------------------')
            self.train_step(train_ds, optimizer=optimizer, epoch_num=epoch + 1)
            if val_ds is not None:
                self.validate_step(val_ds, epoch + 1)

    def predict(self, inputs: torch.tensor, number2label=None, mask=None):
        """

        :param inputs: 输入样本数据 shape:(sample_size, time_step, feature_size)
        :param number2label: 索引对应标签名，若不为空，则返回具体类别名字 type:list
        :param mask :(sample_size, time_step)
        :return:[[label1, label2,...], [label1, label2,...], ...]  label为字符串或者数字索引
        """
        self.train(False)
        if mask:
            seq_lens = torch.sum(mask, dim=-1)  # (sample_size)
        else:
            length = inputs.shape[0]
            seq_lens = torch.tensor([self.time_step for _ in range(length)], dtype=torch.int64)
        outputs = self.forward(inputs)
        y_pre_list = []
        if number2label is not None:
            if isinstance(number2label, list):
                for i, sample in enumerate(outputs):
                    y_pre = torch.argmax(sample[:seq_lens[i], :], dim=-1)  # seq_len
                    # print('predict y is:', y_pre)
                    labels = []
                    for j in range(y_pre.shape[0]):
                        labels.append(number2label[y_pre[j].item()])
                    y_pre_list.append(labels)
            else:
                return None
        else:
            for i, sample in enumerate(outputs):
                y_pre = torch.argmax(sample[:seq_lens[i], :], dim=-1)
                labels = []
                for j in range(y_pre.shape[0]):
                    labels.append(y_pre[i].item())
                y_pre_list.append(labels)
        return y_pre_list

    def test(self, x_test, y_test):
        """
        评估测试集上的表现
        :return:
        """
        self.train(False)
        data_loader = DataLoader(BlockData(x=x_test, y=y_test, feature_size=self.feature_size, time_step=self.time_step,
                                           type='rnn'), shuffle=True, collate_fn=BaseModel.collate_fn,
                                 batch_size=self.batch_size)
        self.validate_step(inputs=data_loader, test_data=True)

    @staticmethod
    def collate_fn(batch):
        xs, ys, mask = [], [], []
        # print(batch)
        max_len = 12
        for (x, y, length) in batch:
            xs.append(x)
            ys.append(y)
            tmp = [0 for _ in range(max_len)]
            for i in range(length):
                tmp[i] = 1
            mask.append(tmp)
        xs = torch.tensor(xs, dtype=torch.float32).cuda()
        # print('after xs')
        # print(ys)
        # print(mask)
        ys = torch.tensor(ys, dtype=torch.int64).cuda()
        # print('after ys')
        mask = torch.tensor(mask, dtype=torch.int64).cuda()
        # print('after mask')
        # (batch_size, time_step, feature_size), (batch_size, time_step), (batch_size, time_step)
        # print('collate_fn:', xs.shape, ys.shape, mask.shape)
        return xs, ys, mask
