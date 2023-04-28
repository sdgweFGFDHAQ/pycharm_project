# coding=utf-8

# Lint as: python3
"""BERT-based LaserTagger runner."""
import logging
import os

import pandas as pd
from absl import flags
from torch.utils.data import Dataset, DataLoader
from model_operate import ModelFnBuilder
import process_data
from workplace.laser_tagger import utils
from workplace.fewsamples.utils.mini_tool import cut_word
FLAGS = flags.FLAGS

# Required parameters
source_data_path = '../all_labeled_data.csv'
train_file = './dataset/train.csv'
eval_file = './dataset/eval.csv'
test_file = './dataset/test.csv'
label_map_file = '/home/data/temp/zzx/lasertagger-chinese/output/label_map.txt'

# Other parameters
# Initial checkpoint, usually from a pre-trained BERT model
init_checkpoint = '/home/data/temp/zzx/lasertagger-chinese/bert_base/RoBERTa-tiny-clue/bert_model.ckpt'
max_seq_length = 7
hidden_dim = 200
train_batch_size = 32
batch_size = 16
# The initial learning rate for Adam
learning_rate = 3e-5
num_train_epochs = 3
dropout = 0.5
# How many steps to make in each estimator call
iterations_per_loop = 200
num_train_examples = 3000
num_eval_examples = 1000

# Path to save the exported model
export_path = '/home/data/temp/zzx/lasertagger-chinese/models/cefect/export'


def get_standard_data(source_path, target_path, number, is_concat=True):
    source_df = pd.read_csv(source_path)
    phrase_list = set()
    source_list, target_list = [], []
    csv2txt = target_path.replace('.csv', '.txt')
    if is_concat:
        for _, df in source_df.groupby('category3_new'):
            for _ in range(number):
                sample1 = df.sample(n=1, random_state=None)
                text_1 = (sample1['name'].values + sample1['category3_new'].values)[0]
                sample2 = df.sample(n=1, random_state=None)
                text_2 = (sample2['name'].values + sample2['category3_new'].values)[0]
                phrase_list.add(text_1 + '[seq]' + text_2)
        for i in phrase_list:
            il = i.split('[seq]')
            source_list.append(il[0])
            target_list.append(il[1])
        pd.DataFrame({'source': source_list, 'target': target_list}).to_csv(target_path)
    else:
        for _, df in source_df.groupby('category3_new'):
            for _ in range(number):
                sample1 = df.sample(n=1, random_state=None)
                text_1 = (sample1['name'].values + sample1['category3_new'].values)[0]
                phrase_list.add(text_1)
        for i in phrase_list:
            source_list.append(i[0])
        pd.DataFrame({'source': source_list}).to_csv(target_path)
    mode = 'w' if os.path.exists(csv2txt) else 'a'
    with open(csv2txt, mode=mode, encoding='utf-8') as f:
        f.writelines("%s\n" % p for p in phrase_list)



class DefineDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __getitem__(self, index):
        if self.label is None:
            return self.data[index]
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


def run_main():
    
    process_data.phrase_vocabulary()

    csv_df = pd.read_csv(train_file)
    train_ds = DefineDataset(csv_df['source'].values, csv_df['target'].values)
    train_ip = DataLoader(dataset=train_ds, batch_size=train_batch_size, shuffle=True, drop_last=True)

    csv_df = pd.read_csv(eval_file)
    eval_ds = DefineDataset(csv_df['source'].values, csv_df['target'].values)
    eval_ip = DataLoader(dataset=eval_ds, batch_size=train_batch_size, shuffle=True, drop_last=True)

    # csv_df = pd.read_csv(test_file)
    # test_ds = DefineDataset(csv_df['source'].values, csv_df['target'].values)
    # test_ip = DataLoader(dataset=test_ds, batch_size=train_batch_size, shuffle=True, drop_last=True)

    num_tags = len(utils.read_label_map(label_map_file))
    mfb = ModelFnBuilder(
        num_tags=num_tags,
        hidden_dim=hidden_dim,
        lt_dropout=dropout,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length)
    final_hidden, model = mfb.pretrain_model(train_ip, model_name='bert-base-chinese')
    mfb.training(train_ip, model)
    mfb.evaluate(eval_ip, model)
    # operate_model.predict(test_ip, model)


if __name__ == '__main__':
    get_standard_data(source_data_path, train_file, 200)
    get_standard_data(source_data_path, eval_file, 20)
    get_standard_data(source_data_path, test_file, 50)
    # run_main()
