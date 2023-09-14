import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

num_epochs = 10


# 定义模型M1
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.fc = nn.Linear(10, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x


# 定义模型M2
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.fc = nn.Linear(10, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x


# 定义模型M3
class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.fc = nn.Linear(10, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x


# 定义适配器类将PyTorch模型转换为sklearn的可调用对象
class PyTorchAdapter:
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        inputs = torch.tensor(x)
        outputs = self.model(inputs)
        _, predicted_labels = torch.max(outputs, 1)
        return predicted_labels.numpy()


# 定义Stacking集成模型
class StackingModel(nn.Module):
    def __init__(self):
        super(StackingModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def ads():
    # 加载数据集
    X, y = load_iris(return_X_y=True)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 将PyTorch模型转换为sklearn的可调用对象
    model1 = Model1()
    model2 = Model2()
    model3 = Model3()

    # 创建适配器对象
    model1_adapter = Model1()
    model2_adapter = Model2()
    model3_adapter = Model3()

    # 定义投票集成模型
    voting_model = VotingClassifier(
        estimators=[('model1', model1_adapter), ('model2', model2_adapter), ('model3', model3_adapter)],
        voting='hard'  # 'soft'表示使用软投票
    )

    # 训练投票集成模型
    voting_model.fit(X_train, y_train)

    # 对测试集进行预测
    voting_predictions = voting_model.predict(X_test)

    # 计算投票集成模型的准确率
    voting_accuracy = accuracy_score(y_test, voting_predictions)
    print("Voting Accuracy:", voting_accuracy)

    # 定义AdaBoost集成模型
    adaboost_model = AdaBoostClassifier(
        base_estimator=LogisticRegression(),
        n_estimators=3
    )

    # 训练AdaBoost集成模型
    adaboost_model.fit(X_train, y_train)

    # 对测试集进行预测
    adaboost_predictions = adaboost_model.predict(X_test)

    # 计算AdaBoost集成模型的准确率
    adaboost_accuracy = accuracy_score(y_test, adaboost_predictions)
    print("AdaBoost Accuracy:", adaboost_accuracy)

    stacking_model = StackingModel()
    stacking_optimizer = optim.Adam(stacking_model.parameters(), lr=0.001)
    stacking_criterion = nn.CrossEntropyLoss()

    # 训练Stacking由于Stacking方法需要在训练集上训练基模型，并使用基模型的预测结果作为输入训练Stacking模型，因此需要额外的训练步骤。以下是Stacking方法的代码示例：

    # 创建基模型列表
    base_models = [model1, model2, model3]
    base_model_predictions_train = []
    base_model_predictions_test = []

    # 在训练集上训练基模型，并获取其预测结果
    for base_model in base_models:
        base_model.train()
        optimizer = optim.Adam(base_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = base_model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        base_model.eval()
        train_predictions = base_model.predict(X_train)
        test_predictions = base_model.predict(X_test)

        base_model_predictions_train.append(train_predictions)
        base_model_predictions_test.append(test_predictions)

    # 转换预测结果为numpy数组
    base_model_predictions_train = np.array(base_model_predictions_train).T
    base_model_predictions_test = np.array(base_model_predictions_test).T

    # 将基模型的预测结果与原始特征拼接
    X_train_stacked = np.concatenate((X_train, base_model_predictions_train), axis=1)
    X_test_stacked = np.concatenate((X_test, base_model_predictions_test), axis=1)

    # 训练Stacking模型
    stacking_model = StackingModel()
    stacking_optimizer = optim.Adam(stacking_model.parameters(), lr=0.001)
    stacking_criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        stacking_optimizer.zero_grad()
        outputs = stacking_model(X_train_stacked)
        loss = stacking_criterion(outputs, y_train)
        loss.backward()
        stacking_optimizer.step()

    # 在测试集上进行预测
    stacking_model.eval()
    stacking_predictions = stacking_model(X_test_stacked)

    # 计算Stacking模型的准确率
    _, stacking_predicted_labels = torch.max(stacking_predictions, 1)
    stacking_accuracy = accuracy_score(y_test, stacking_predicted_labels.numpy())
    print("Stacking Accuracy:", stacking_accuracy)


if __name__ == '__main__':
    ads()
