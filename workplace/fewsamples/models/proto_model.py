# encoding=utf-8
import torch
from torch import nn
import torch.nn.functional as F
from icecream.icecream import ic
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ProtoTypicalNet(nn.Module):
    def __init__(self, bert_layer, input_dim, hidden_dim, num_class, dropout=0.5, requires_grad=False):
        super(ProtoTypicalNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class

        # 线性层进行编码
        self.bert_embedding = bert_layer
        for param in self.bert_embedding.parameters():
            param.requires_grad = requires_grad

        self.prototype = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_class)
        )

    def forward(self, support_input, query_input):
        # # 由于版本原因，当前选择的bert模型会返回tuple，包含(last_hidden_state,pooler_output)
        support_embedding = self.bert_embedding(support_input).last_hidden_state[:, 0]
        query_embedding = self.bert_embedding(query_input).last_hidden_state[:, 0]

        # 距离即loss
        support_size0 = support_embedding.shape[0]
        every_class_num = support_size0 // self.num_class
        class_meta_dict = {}
        for i in range(0, self.num_class):
            class_meta_dict[i] = torch.sum(support_embedding[i * every_class_num:(i + 1) * every_class_num, :],
                                           dim=0) / every_class_num

        class_meta_information = torch.zeros(size=[len(class_meta_dict), support_embedding.shape[1]])
        for key, item in class_meta_dict.items():
            class_meta_information[key, :] = class_meta_dict[key]

        N_query = query_embedding.shape[0]
        result = torch.zeros(size=[N_query, self.num_class])
        for i in range(0, N_query):
            temp_value = query_embedding[i].repeat(self.num_class, 1)
            cosine_value = torch.cosine_similarity(class_meta_information, temp_value, dim=1)
            result[i] = cosine_value

        result = self.prototype(support_embedding)
        ic(result)
        return result
