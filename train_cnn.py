import os
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import torch.nn as nn
from torch import optim
import numpy as np
from model.LSS_LSTM import LSS_LSTM
from utils import drawAUC_TwoClass, plot_and_save, calculate_f1_score, calculate_auc, calculate_mse_rmse
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from scipy.ndimage import rotate

import config

config = config.config


# train函数
def train(alldata_train, alltarget_train, alldata_val, alltarget_val):
    # 最大加权综合分数
    max_score = 0
    # 读取训练数据集配置dataset
    train_dataset = TensorDataset(torch.from_numpy(alldata_train).float(),torch.from_numpy(alltarget_train).float())
    # 使用dataset和batch_size配置datasetloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True)
    # 读取验证数据集配置dataset
    val_dataset = TensorDataset(torch.from_numpy(alldata_val).float(), torch.from_numpy(alltarget_val).float())
    # 使用dataset和batch_size配置datasetloader
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=config["batch_size"], shuffle=True)
    # 读取模型送至指定GPU
    model = LSS_LSTM().to(config["device"])

    model_name = model.__class__.__name__
    ###  设置保存文件夹命名格式   ###
    if "raw" in config["newdata_path"]:
        Data_Type = "Raw"
    else:
        Data_Type = "FR"
    if config["normalize"] and not config["normalize_to_0_1"]:
        result_folder = f"{model_name}_std_" + Data_Type
    elif not config["normalize"] and config["normalize_to_0_1"]:
        result_folder = f"{model_name}_0_1_" + Data_Type
    elif config["normalize"] and config["normalize_to_0_1"]:
        raise ValueError("config['normalize'] 和 config['normalize_to_0_1'] 不能同时为 True")
    else:
        result_folder = f"{model_name}_" + Data_Type

    writer = SummaryWriter(log_dir=os.path.join('Result', result_folder, 'log_dir'))

    # 定义损失函数和优化器
    # 交叉熵损失
    criterion = nn.CrossEntropyLoss().to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    train_acc_list = []
    train_loss_list = []
    train_f1_list = []
    val_acc_list = []
    val_loss_list = []
    val_f1_list = []

    for epoch in range(config["epochs"]):
        train_acc = 0.0
        train_loss = 0.0
        train_f1 = 0.0
        train_mse = 0.0
        train_rmse = 0.0
        val_acc = 0.0
        val_loss = 0.0
        val_f1 = 0.0
        val_mse = 0.0
        val_rmse = 0.0
        train_outputs_list = []
        train_labels_list = []
        val_outputs_list = []
        val_labels_list = []

        # 指定模型进入训练模式
        model.train()
        for images, target in train_loader:
            # 数据和标签送入指定GPU
            images, target = Variable(images).to(config["device"]), Variable(target).to(config["device"])
            # 清除梯度
            optimizer.zero_grad()
            # 模型预测结果
            outputs = model(images)
            # 获得预测值最大索引
            _, preds = torch.max(outputs.data, 1)
            # 计算损失函数
            loss = criterion(outputs, target.squeeze().long())
            # 反向传播
            loss.backward()
            # 优化器更新模型参数
            optimizer.step()
            train_outputs_list.extend(outputs.detach().cpu().numpy())
            train_labels_list.extend(target.cpu().numpy())
            train_array = np.array(train_outputs_list)
            train_mse, train_rmse = calculate_mse_rmse(outputs, target)
            train_acc += (preds[..., None] == target).squeeze().sum().cpu().numpy()
            train_loss += loss.item()
            # 计算F1分数
            train_f1 = calculate_f1_score(target.cpu().numpy(), preds.cpu().numpy())
            # 计算AUC，AUC为ROC曲线下的面积
            train_auc = calculate_auc(train_labels_list, train_array[:, 1])  

        writer.add_scalars('LOSS/', {'Train_Loss': train_loss / len(train_dataset)}, epoch)
        writer.add_scalars('ACC/', {'Train_Acc': float(train_acc) / len(train_dataset)}, epoch)

        # 指定模型进入评价模式，不计算梯度
        model.eval()
        # 不计算梯度
        with torch.no_grad():
            for images, target in val_loader:
                # 数据和标签送入指定GPU
                images, target = Variable(images).to(config["device"]), Variable(target).to(config["device"])
                # 模型预测结果
                outputs = model(images)
                # 计算损失函数
                loss = criterion(outputs, target.squeeze().long())
                val_loss += loss.item()
                val_outputs_list.extend(outputs.detach().cpu().numpy())
                val_labels_list.extend(target.cpu().numpy())
                score_array = np.array(val_outputs_list)
                val_mse, val_rmse = calculate_mse_rmse(outputs, target)
                # 获得预测值最大索引
                _, preds = torch.max(outputs.data, 1)
                val_acc += (preds[..., None] == target).squeeze().sum().cpu().numpy()
                # 计算F1分数
                val_f1 = calculate_f1_score(target.cpu().numpy(), preds.cpu().numpy())
                # 计算AUC
                val_auc = calculate_auc(val_labels_list, score_array[:, 1])
            # 计算加权综合分数
            weight_score = 0.2 * (val_acc / len(val_dataset)) + 0.6 * (val_auc / 100) + 0.2 * val_f1

            print('[%03d/%03d]  Train Acc: %3.5f AUC: %3.5f Loss: %3.6f F1: %3.5f MSE: %3.5f RMSE: %3.5f'
                  '| Val Acc: %3.5f AUC: %3.5f Loss: %3.6f F1: %3.5f MSE: %3.5f RMSE: %3.5f'
                  '| Weight Score: %3.5f' % \
                  (epoch + 1, config["epochs"], train_acc / len(train_dataset), train_auc,
                   train_loss / len(train_dataset), train_f1, train_mse, train_rmse,
                   val_acc / len(val_dataset), val_auc, val_loss / len(val_dataset), val_f1, val_mse, val_rmse, weight_score))

            # save result for each epoch
            file_path = os.path.join('Result', result_folder, f"{result_folder}.txt")
            with open(file_path, 'a') as f:
                f.write('[%03d/%03d]  Train Acc: %3.5f AUC: %3.5f Loss: %3.6f F1: %3.5f MSE: %3.5f RMSE: %3.5f'
                        '| Val Acc: %3.5f AUC: %3.5f Loss: %3.6f F1: %3.5f MSE: %3.5f RMSE: %3.5f'
                        '| Weight Score: %3.5f\n' % \
                        (epoch + 1, config["epochs"], train_acc / len(train_dataset), train_auc,
                         train_loss / len(train_dataset), train_f1, train_mse, train_rmse,
                         val_acc / len(val_dataset), val_auc, val_loss / len(val_dataset), val_f1, val_mse, val_rmse, weight_score))
            train_acc_list.append(train_acc / len(train_dataset))
            train_loss_list.append(train_loss / len(train_dataset))
            train_f1_list.append(train_f1)
            val_acc_list.append(val_acc / len(val_dataset))
            val_loss_list.append(val_loss / len(val_dataset))
            val_f1_list.append(val_f1)
            # 保留最大的加权综合分数结果，绘制对应的训练AUC曲线和验证AUC曲线，保存对应的模型参数
            if weight_score > max_score:
                max_score = weight_score
                drawAUC_TwoClass(train_labels_list, train_array[:, 1],
                                    os.path.join('Result', result_folder, 'train_AUC.png'))
                drawAUC_TwoClass(val_labels_list, score_array[:, 1],
                                    os.path.join('Result', result_folder, 'val_AUC.png'))
                train_auc = calculate_auc(train_labels_list, train_array[:, 1])  # auc为Roc曲线下的面积
                val_auc = calculate_auc(val_labels_list, score_array[:, 1])
                best_result = os.path.join('Result', result_folder, "best_result.txt")
                with open(best_result, 'w') as f:
                    f.write('Train Acc: %3.5f AUC: %3.5f Loss: %3.6f F1: %3.5f MSE: %3.5f RMSE: %3.5f'
                            '| Val Acc: %3.5f AUC: %3.5f Loss: %3.6f F1: %3.5f MSE: %3.5f RMSE: %3.5f'
                            '| Weight Score: %3.5f' % \
                            (train_acc / len(train_dataset), train_auc, train_loss / len(train_dataset), train_f1,
                                train_mse, train_rmse,
                                val_acc / len(val_dataset), val_auc, val_loss / len(val_dataset), val_f1, val_mse,
                                val_rmse, weight_score))
                torch.save(model.state_dict(), os.path.join('Result', result_folder, 'best.pth'))
            # 记录Loss, accuracy
            writer.add_scalars('LOSS/valid', {'valid_loss': val_loss / len(val_dataset)}, epoch)
            writer.add_scalars('ACC/valid', {'valid_acc': val_acc / len(val_dataset)}, epoch)
    # 绘制acc、loss、f1分数曲线
    plot_and_save(os.path.join('Result', result_folder), train_acc_list, train_loss_list,
                  train_f1_list, val_acc_list, val_loss_list, val_f1_list)
    # 保存模型的最新参数
    torch.save(model.state_dict(), os.path.join('Result', result_folder, 'latest.pth'))
