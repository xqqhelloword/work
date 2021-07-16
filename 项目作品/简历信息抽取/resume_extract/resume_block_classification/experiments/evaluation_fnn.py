import torch
from constant.special_string import *
from resume_block_classification.experiments.base_model import BaseModel
from resume_block_classification.data_precess.load_resume_data import load_data_for_rnn_new
from resume_block_classification.data_precess.feature_engineer import trans_to_wordvec_by_word2vec
from resume_block_classification.data_precess.create_data import number2label
from resume_segmentation.resume_segement import segment_one_resume_from_file


class FNNModel(BaseModel):
    def __init__(self, time_step, feature_size, optimizer=torch.optim.Adam, batch_size=5, epochs=10):
        super(FNNModel, self).__init__(time_step=time_step, feature_size=feature_size,
                                       batch_size=batch_size, epochs=epochs)
        self.linear1 = torch.nn.Linear(in_features=feature_size, out_features=64)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(in_features=64, out_features=64)
        self.relu2 = torch.nn.ReLU()
        self.linear_fnn = torch.nn.Linear(in_features=64, out_features=10)
        self.drop = torch.nn.Dropout(0.2)
        self.batch_norm1 = torch.nn.BatchNorm1d(num_features=64, affine=True)
        self.optimizer = optimizer(params=self.parameters())

    def forward(self, x):
        """

        :param x: (sample_num, time_step, item_d)
        :return:
        """
        # print(x.shape)
        outputs_fnn = []
        for i in range(x.shape[1]):
            out_fnn = self.linear1(x[:, i, :])  # (sample_size, 64)  # fnn
            out_fnn = self.relu1(out_fnn)
            out_fnn = self.batch_norm1(out_fnn)
            out_fnn = self.linear2(out_fnn)  # out_shape:64
            out_fnn = self.drop(out_fnn)
            out_fnn = self.relu2(out_fnn)
            out_fnn = self.linear_fnn(out_fnn)
            outputs_fnn.append(out_fnn)
            # print('out_fnn.shape, final:', out_fnn.shape)
        return torch.stack(outputs_fnn, dim=1)  # 最终结果shape:(batchsz, time_step, 10)


if __name__ == '__main__':
    def evaluation_fnn():
        train_x, train_y, test_x, test_y, val_x, val_y = load_data_for_rnn_new(data_set=3,
                                                                               train_num=700, test_num=400, val_num=100)
        # print(val_y)
        brnn = FNNModel(time_step=12, feature_size=100, epochs=10, batch_size=10)
        brnn.to('cuda')

        brnn.fit(train_x, train_y, optimizer=brnn.optimizer)
        brnn.test(test_x, test_y)
        # print(brnn.layers)

    def test_model():
        file_path = 'D:\\Download\\简历模板.docx'
        brnn_save = FNNModel(time_step=12, feature_size=100, epochs=10, batch_size=5)
        brnn_save.load_weights(FNN_MODEL_PATH)
        one_resume = segment_one_resume_from_file(file_path)
        for module in one_resume:
            print(module)
            print('-------------------------------------------------------------------------')
        text_list = [one_resume]
        inputs = trans_to_wordvec_by_word2vec(text_list, feature_size=100,
                                              word2vec_model=word2vec_model_path_2021_4_2, type='rnn',
                                              time_step=brnn_save.time_step)

        print(brnn_save.predict(inputs), brnn_save.predict(inputs, number2label=number2label))

    evaluation_fnn()
    # test_model()

