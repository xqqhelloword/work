
import gensim
from torch.utils.data import Dataset
from constant.special_string import *
from typing import List
from resume_block_classification.data_precess.feature_engineer import trans_to_wordvec_by_word2vec, process_rnn_label_list


class BlockData(Dataset):

    def __init__(self, x: List[List[str]], y: List, feature_size, type='rnn', time_step=12):
        super(BlockData, self).__init__()
        word2vec_model = gensim.models.Word2Vec.load(word2vec_model_path_2021_4_2)
        self.x = trans_to_wordvec_by_word2vec(text_list=x, feature_size=feature_size, type=type,
                                              time_step=time_step, word2vec_model=word2vec_model)
        self.seq_len = []
        for one_resume in y:
            length = min(len(one_resume), time_step)
            self.seq_len.append(length)
        self.label_distribution: dict = process_rnn_label_list(label_list=y, time_step=time_step)
        self.y = y

    def __getitem__(self, index):
        x_i, y_i, len_module = self.x[index], self.y[index], self.seq_len[index]
        return x_i, y_i, len_module

    def __len__(self):
        return len(self.y)
