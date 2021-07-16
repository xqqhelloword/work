import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, n_class=5, num_2_tag=None, tag_2_num=None, gpu=True):
        super(CRF, self).__init__()
        self.n_class = n_class
        self.gpu = gpu
        self.transform = nn.Parameter(torch.randn((n_class+2, n_class+2), requires_grad=True))  # 状态转移矩阵
        self.num_2_tag = ['B', 'I', 'O', 'S', 'T', 'START', 'END'] \
            if num_2_tag is None else num_2_tag.extend(['START', 'END'])
        self.tag_2_num = {'B': 0, 'I': 1, 'O': 2, 'S': 3, 'T': 4, 'START': 5, 'END': 6} \
            if tag_2_num is None else tag_2_num.update({'START': 5, 'END': 6})
        # self.layer_norm_trans = nn.LayerNorm([n_class, n_class])

    def forward(self, x, mask=None):
        # crf层的输出是维特比解码后的标签序列
        # x.shape:(batch_size, time_step, n_class)
        with torch.no_grad():
            res = []
            seq_lens = torch.sum(mask, dim=1) if mask is not None else torch.tensor[x.shape[1]].expand([x.shape[0]])
            # print(seq_lens)
            for i in range(x.shape[0]):
                res.append(self.viterbi_decode_one_sample(torch.transpose(x[i, :seq_lens[i].item(), :], dim0=0, dim1=1)))
        return res

    def viterbi_decode_one_sample(self, feats):
        # feats.shape:(n_class, time_step), feats:状态特征分数矩阵，即对应CRF层的输入，通常为RNN模型的输出结果
        # 解码思想：得分和最大的路径则为解码后的真实标签索引序列，采用动态规划方式
        n_class = feats.shape[0]
        seq_len = feats.shape[1]
        with torch.no_grad():
            assert seq_len > 0
            dp = [feats[i, 0] + self.transform[self.tag_2_num['START'], i] for i in range(n_class)]  # 初始得分
            path = torch.zeros((n_class, seq_len), dtype=torch.int64).cuda()
            path[:, 0] = self.tag_2_num['START']  # 存储真实路径标签(当前时间步的上一个标签索引)，初始时刻的标签路径为-1，表示在这之前已经没有上一个节点
            dp = torch.tensor(dp).cuda() if self.gpu else torch.tensor(dp)  # shape:(n_class)
            for i in range(1, seq_len):  # 递推
                new_state = torch.max(dp.expand(n_class, n_class).t() + self.transform[:n_class, :n_class], dim=0)  # (values, indexes)
                dp, path[:, i] = new_state[0] + feats[:, i], new_state[1]  # (n_class)
            max_score = torch.max(dp + self.transform[:n_class, self.tag_2_num['END']]).item()
            last_index = torch.argmax(dp + self.transform[:n_class, self.tag_2_num['END']]).item()
            ans_path = []
            i = feats.shape[1] - 1  # 最后一个时间步索引
            # print('path:', path)
            while last_index != self.tag_2_num['START'] and i >= 0:
                ans_path.append(last_index)
                last_index = path[last_index, i].item()
                i -= 1
        real_path = ans_path[::-1]
        tag_seq = [self.num_2_tag[index] for index in real_path]
        return max_score, ans_path[::-1], tag_seq

    def neg_log_likelihood_loss(self, x, target, mask=None):
        """
        计算负对数似然损失
        x.shape:(batch_size, time_step, n_class)
        target.shape:(batch_size, time_step)
        :param x:
        :param mask:
        :param target:
        :return:
        """
        loss = 0
        # seq_lens = torch.sum(mask, dim=1)  # 存储了样本的真实序列长度 shape:(batch_size)
        seq_lens = torch.sum(mask, dim=1) if mask is not None else torch.tensor[x.shape[1]].expand([x.shape[0]])
        for i, lens in enumerate(seq_lens):  # 存储了每一个样本的真实序列长度
            loss += self.one_sample_neg_log_likelihood_loss(x[i, :lens, :].t(), target[i, :lens])
        return loss

    def one_sample_neg_log_likelihood_loss(self, input, target):
        """
        计算单个样本的真实标签序列的负对数似然损失
        :param input: (n_class, time_step)
        :target: 输出序列(0,1,2,...) shape:(seq_len) 1维张量
        :return:
        """
        # print('input.shape:', input.shape)
        # print('target.shape:', target.shape)
        n_class = input.shape[0]
        seq_len = target.shape[0]
        # print(input)
        # self.transform = self.layer_norm_trans(self.transform)
        score_x_y = input[target[0].item(), 0] + self.transform[self.tag_2_num['START'], target[0].item()]  # 计算序列target的得分
        for i in range(1, seq_len):
            score_x_y += input[target[i].item(), i]
            # print('input[target[i].item(), i]:', input[target[i].item(), i])
            score_x_y += self.transform[target[i-1].item(), target[i].item()]
        score_x_y += self.transform[target[-1].item(), self.tag_2_num['END']]
        # print('score of sequence:', score_x_y)
        # print(self.transform)
        Z_t = input[:, 0] + self.transform[self.tag_2_num['START'], :n_class]
        for i in range(1, target.shape[0]):  # dp递推计算
            tmp = Z_t.expand(n_class, n_class).t() + self.transform[:n_class, :n_class] + input[:, i].expand(n_class, n_class)
            max_val, _ = torch.max(tmp, dim=0)
            Z_t = torch.log(torch.sum(torch.exp(tmp-max_val))) + max_val
        tmp = Z_t + self.transform[:n_class, self.tag_2_num['END']]
        max_val = torch.max(tmp)
        total_score = torch.log(torch.sum(torch.exp(tmp-max_val))) + max_val
        # print('total_score:', total_score)
        return total_score - score_x_y  # 负对数似然loss



