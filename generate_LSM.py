import os
import time
import pickle
import torch
import read_data
import numpy as np
from torch.autograd import Variable
from model.LSS_LSTM import LSS_LSTM
from tqdm import tqdm
from utils import plot_save_lsm
from torch.utils.data import DataLoader, TensorDataset
import config

config = config.config


def save_LSM():
    # 是否要绘画并保存LSM的png图片
    plot_and_save = True
    print('*******************************************生成LSM*******************************************')
    # 读取模型送至指定GPU
    model = LSS_LSTM().to(config["device"])
    # 读取指定路径模型参数
    model.load_state_dict(torch.load(os.path.join('Result', 'LSS_LSTM_0_1_FR', 'best.pth')))
    # 读取训练数据 feature*height*width
    tensor_data = read_data.get_feature_data()
    print('整个预测区域大小：' + str(tensor_data.shape))
    creat = read_data.creat_dataset(tensor_data)
    data = creat.creat_new_tensor()
    images_list = []
    probs = []
    # 索引方便转换数据格式
    index_array = np.array([[0, 0], [0, 1], [0, 2], [1, 2], [1, 1], [1, 0], [2, 0], [2, 1], [2, 2]])
    # # 指定模型进入评价模式，不计算梯度
    model.eval()
    with torch.no_grad():   
        # 遍历有效数据区域（除去扩大的边缘）
        for i in range(1, config["height"] + 1):
            for j in range(1, config["width"] + 1):
                # 读取若干个 3 × 3 × 13 区域
                images_list.append(data[:, i - 1:i + 2, j - 1:j + 2].astype(np.float32))
                if (i != 1 and (i - 1) % config["Cutting_window"] == 0 and (j - 1)== config["width"] - 1) or ( 
                    (i - 1) == config["height"] - 1 and (j - 1) == config["width"] - 1):
                    start_time = time.time()
                    print('i=' + str(i) + ' j=' + str(j))
                    # 将每一个块按照 S 型排列将 3 × 3 × 13 转换为 9 × 13
                    pred_data = np.array([np.array([image[:, index_array[i, 0], index_array[i, 1]] for i in range(9)]) for image in images_list])
                    images_list = []
                    pred_dataset = TensorDataset(torch.from_numpy(pred_data))
                    pred_loader = DataLoader(dataset=pred_dataset, batch_size=config["batch_size"], shuffle=False)
                    # tqdm相当于给过程添加进度条
                    for images in tqdm(pred_loader):
                        # 整合、转换数据格式方便丢入模型
                        images = torch.stack([image.cuda() for image in images])
                        # 调整数据形状并将数据转移到指定GPU上
                        images = Variable(images.squeeze(0)).to(config["device"])
                        # 将数据丢入模型得到预测为滑坡的概率
                        probs.append(model(images).cpu()[:, 1])
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    minutes = int(elapsed_time // 60)
                    seconds = elapsed_time % 60
                    print(f"滑动窗口处理时间: {minutes} 分钟 {seconds} 秒")
    # 拼接概率列表方便转换为地图格式
    probs = np.concatenate(probs)
    print('概率列表生成完成')
    if plot_and_save:
        # 绘画并保存LSM的png图片
        plot_save_lsm(os.path.join('Result', 'LSS_LSTM_0_1_FR'), probs)
    # 保存为tif文件
    read_data.save_to_tif(probs, os.path.join('Result', 'LSS_LSTM_0_1_FR', 'LSTM_LSM.tif'))
            