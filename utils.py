import math
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score
import torch.nn.functional as F
from torch import nn
import config

config = config.config

def calculate_mse_rmse(outputs, target):
    outputs_softmax = F.softmax(outputs, dim=1)
    target_one_hot = F.one_hot(target.long(), num_classes=2).squeeze()
    mse_loss = F.mse_loss(outputs_softmax, target_one_hot.float())
    rmse = torch.sqrt(mse_loss)
    return mse_loss.item(), rmse.item()


# AUC绘制函数
def drawAUC_TwoClass(y_true, y_score, path):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)  # auc为Roc曲线下的面积
    roc_auc = roc_auc * 100
    # 开始画ROC曲线
    plt.figure(figsize=(5, 5), dpi=300)
    plt.plot(fpr, tpr, color='darkorange', linestyle='-', linewidth=2,
             label=('CNN (' + str(path).split('.')[0] + ' = %0.2f %%)' % roc_auc))
    plt.legend(loc='lower right')  # 设定图例的位置，右下角
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.tick_params(direction='in', top=True, bottom=True, left=True, right=True)  # 坐标轴朝向
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xticks(np.arange(0, 1.1, 0.1))

    plt.grid(linestyle='-.')
    plt.xlabel('False Positive Rate')  # 横坐标是fpr
    plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    plt.legend(loc="lower right")

    # print("AUC:",roc_auc)
    plt.savefig(path, format='png')
    plt.close()


def calculate_f1_score(y_true, y_pred):
    """
    计算F1分数

    参数:
    y_true : 真实标签列表
    y_pred : 预测标签列表

    返回:
    f1 : F1分数
    """
    f1 = f1_score(y_true, y_pred, average='macro')
    return f1


def calculate_auc(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr) * 100  # auc为Roc曲线下的面积
    return roc_auc


def plot_and_save(path, train_acc_list, train_loss_list, train_f1_list, val_acc_list, val_loss_list, val_f1_list):
    epochs = range(1, len(train_acc_list) + 1)
    # 绘制曲线并保存为 png 图片
    plt.clf()
    plt.figure(figsize=(7, 5), dpi=200)
    plt.subplots_adjust(left=0.3)
    plt.plot(epochs, train_acc_list, color='darkorange', linestyle='-', linewidth=2, label='Train Acc')
    plt.legend(loc='lower right')  # 设定图例的位置，右下角
    plt.xlabel('Epoch')
    plt.ylabel('Train Acc')
    plt.savefig(os.path.join(path, 'train_Acc.png'), format='png')

    plt.clf()
    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(epochs, train_loss_list, color='darkorange', linestyle='-', linewidth=2, label='Train Loss')
    plt.legend(loc='upper right')  # 设定图例的位置，右下角
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.savefig(os.path.join(path, 'train_Loss.png'), format='png')

    plt.clf()
    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(epochs, train_f1_list, color='darkorange', linestyle='-', linewidth=2, label='Train F1')
    plt.legend(loc='upper right')  # 设定图例的位置，右下角
    plt.xlabel('Epoch')
    plt.ylabel('Train F1')
    plt.savefig(os.path.join(path, 'train_F1.png'), format='png')

    plt.clf()
    plt.figure(figsize=(7, 5), dpi=200)
    plt.subplots_adjust(left=0.3)
    plt.plot(epochs, val_acc_list, color='darkorange', linestyle='-', linewidth=2, label='Val Acc')
    plt.legend(loc='lower right')  # 设定图例的位置，右下角
    plt.xlabel('Epoch')
    plt.ylabel('Val Acc')
    plt.savefig(os.path.join(path, 'val_Acc.png'), format='png')

    plt.clf()
    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(epochs, val_loss_list, color='darkorange', linestyle='-', linewidth=2, label='Val Loss')
    plt.legend(loc='upper right')  # 设定图例的位置，右下角
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss')
    plt.savefig(os.path.join(path, 'val_Loss.png'), format='png')

    plt.clf()
    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(epochs, val_f1_list, color='darkorange', linestyle='-', linewidth=2, label='Val F1')
    plt.legend(loc='upper right')  # 设定图例的位置，右下角
    plt.xlabel('Epoch')
    plt.ylabel('Val F1')
    plt.savefig(os.path.join(path, 'val_F1.png'), format='png')


def plot_save_lsm(path, probs):
    probs = probs.reshape((config["height"], config["width"]))
    # 数据可视化
    plt.figure(dpi=300)  
    plt.imshow(probs, cmap='hot', interpolation='nearest')
    plt.colorbar()  # 添加颜色条
    plt.savefig(os.path.join(path, 'LSTM_LSM.png'), format='png', dpi=300)  
    plt.show()