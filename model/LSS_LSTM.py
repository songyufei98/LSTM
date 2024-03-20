from torch import nn

class LSS_LSTM(nn.Module):
    def __init__(self):
        super(LSS_LSTM, self).__init__()
        # 13代表输入的每一个时间戳的数据维度（因子个数），25代表隐藏层维度
        self.lstm = nn.LSTM(input_size=13, hidden_size=25, batch_first=True)
        self.bn1 = nn.BatchNorm1d(25)
        # 转换为二分类问题
        self.fc = nn.Linear(25, 2)
        self.bn2 = nn.BatchNorm1d(2)
        # 将结果转换为概率分布
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 只用到x
        x, _ = self.lstm(x)
        # 提取最后一个时间戳的x
        x = self.bn1(x[:, -1, :])
        # 降维转换为二分类问题
        x = self.fc(x)
        x = self.bn2(x)
        # # 将结果转换为概率分布
        x = self.softmax(x)
        return x
