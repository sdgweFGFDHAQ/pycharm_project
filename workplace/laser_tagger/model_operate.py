"""Utilities for building a LaserTagger TF model."""

import torch
from torch import nn, optim
from transformers import BertTokenizer, BertConfig, Trainer, TrainingArguments
from transformers import BertModel, BertForMaskedLM

from laserTagger_model import laserTaggerModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ModelFnBuilder(object):
    """Class for building `model_fn` closure for TPUEstimator."""

    def __init__(self, num_tags, max_seq_length, hidden_dim, lt_dropout, learning_rate):
        """Initializes an instance of a LaserTagger model.

        Args:
          num_tags: Number of different tags to be predicted.
          max_seq_length: Maximum sequence length.
          learning_rate: Learning rate.
        """
        self.num_tags = num_tags
        self.max_seq_length = max_seq_length
        self.hidden_dim = hidden_dim
        self.lt_dropout = lt_dropout
        self.learning_rate = learning_rate

    def pretrain_model(self, inputs, model_name):
        # labels_mask 有效词个数
        """use a Bert model."""
        # a. 通过词典导入分词器
        tokenizer = BertTokenizer.from_pretrained(model_name)
        b = tokenizer(inputs, padding=True, truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        tokens = b.input_ids.to(device)
        # b. 导入配置文件
        model_config = BertConfig.from_pretrained(model_name)
        # 修改配置
        model_config.output_hidden_states = True
        model_config.output_attentions = True
        # 通过配置和路径导入模型
        bert_model = BertModel.from_pretrained(model_name)
        final_hidden = bert_model(tokens).last_hidden_state

        """Creates a LaserTagger model."""
        # 使用普通的dense出概率
        model = laserTaggerModel(
            hidden_dim=self.hidden_dim,
            num_tags=self.num_tags,
            dropout=self.lt_dropout,
        ).to(device)
        return final_hidden, model

    def ssad(self, model, tokenizer, train_dataset, test_dataset):
        training_args = TrainingArguments(
            output_dir='./results',  # 存储结果文件的目录
            overwrite_output_dir=True,
            num_train_epochs=5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=500,
            logging_steps=100,
            seed=2020,
            logging_dir='./logs'  # 存储logs的目录
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=accuracy,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        trainer.train()
        trainer.evaluate()

    def training(self, final_hidden, model):
        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        # 將 model 的模式设定为 train，这样 optimizer 就可以更新 model 的参数
        model.train()
        train_len = len(final_hidden)
        epoch_los, epoch_acc = 0, 0
        for i, (inputs, labels) in enumerate(final_hidden):
            # 1. 放到GPU上
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)
            # 2. 清空梯度
            optimizer.zero_grad()
            # 3. 计算输出
            outputs = model(inputs)
            outputs = outputs.squeeze(1)  # 去掉最外面的 dimension
            # 4. 计算损失
            # outputs:batch_size*num_classes labels:1D
            loss = criterion(outputs, labels)
            epoch_los += loss.item()
            # 5.预测结果
            accu = accuracy(outputs, labels)
            epoch_acc += accu.item()
            # 6. 反向传播
            loss.backward()
            # 7. 更新梯度
            optimizer.step()
        loss_value = epoch_los / train_len
        acc_value = epoch_acc / train_len * 100
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}%'.format(loss_value, acc_value))
        return loss_value, acc_value

    def evaluate(self, val_loader, model):
        criterion = nn.CrossEntropyLoss()
        val_len = len(val_loader)

        model.eval()
        with torch.no_grad():
            epoch_los, epoch_acc = 0, 0
            for i, (inputs, labels) in enumerate(val_loader):
                # 1. 放到GPU上
                inputs = inputs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.long)
                # 2. 计算输出
                outputs = model(inputs)
                outputs = outputs.squeeze(1)
                # 3. 计算损失
                loss = criterion(outputs, labels)
                epoch_los += loss.item()
                # 4. 预测结果
                accu = accuracy(outputs, labels)
                epoch_acc += accu.item()
            loss_value = epoch_los / val_len
            acc_value = epoch_acc / val_len * 100
            print("Valid | Loss:{:.5f} Acc: {:.3f}% ".format(loss_value, acc_value))
        print('-----------------------------------')
        return loss_value, acc_value

    def predict(self, model, pre_input):
        # laserTagger模型
        result = model(pre_input)


def accuracy(pred_y, y):
    pred_list = torch.argmax(pred_y, dim=1)
    correct = (pred_list == y).float()
    acc = correct.sum() / len(correct)
    return acc
