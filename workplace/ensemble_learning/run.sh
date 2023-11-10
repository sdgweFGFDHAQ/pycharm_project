#!/bin/bash

# 使用当前python环境
export python=python
# 系统基础配置
export CUDA_VISIBLE_DEVICES=0

ROOT_PATH='/home/DI/zhouzx/code/workplace/'

LOG_FILE="output.log"
true >"$LOG_FILE"

{ nohup
# 1. 下载文件 训练模型
# prototypical.py
${python} "$ROOT_PATH/fewsamples/prototypical.py" --method count_matching_number;
# 2.预测数据
# prototypical_predict.py
#${python} prototypical_predict.py ;
# 3. 调用集成学习
# 计算均值法
# voting.py
#${python} voting.py ;
# 4. 上传数据
# di_store_drink_label_predict.py
#${python} di_store_drink_label_predict.py ;
} >> "$LOG_FILE" 2>&1

# nohup python -u your_script.py > /dev/null 2>&1 &
