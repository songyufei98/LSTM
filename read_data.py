import os
import warnings
import numpy as np
import pandas as pd
import random
from osgeo import gdal
from PIL import Image
import config
import matplotlib.pyplot as plt

config = config.config
from scipy.ndimage import rotate
import pickle


def normalize(img):
    # 计算均值和标准差
    mean = img.mean()
    std = img.std()
    # 进行标准化
    normalized_img = (img - mean) / std
    return normalized_img


def normalize_to_0_1(img):
    # 计算最小值和最大值
    min_val = img.min()
    max_val = img.max()
    # 进行归一化
    normalized_img = (img - min_val) / (max_val - min_val)
    return normalized_img


def resample_tif(img_re):
    """
    :param img: original factors data
    :return: resampled factors data
    """
    warnings.filterwarnings("ignore")
    img_re = np.array(Image.fromarray(img_re).resize((config["height"], config["width"])))
    return img_re


def read_data_from_tif(tif_path):
    """
    读取影响因子数据并转换为nparray
    """
    tif = gdal.Open(tif_path)
    w, h = tif.RasterXSize, tif.RasterYSize
    img = np.array(tif.ReadAsArray(0, 0, w, h).astype(np.float32))
    if w != config["width"] and h != config["height"]:
        img = resample_tif(img)
    return img


def get_feature_data():
    """"
    读取特征并进行归一化
    """
    tif_paths = config["data_path"]
    data = np.zeros((config["feature"], config["height"], config["width"])).astype(np.float32)
    for i, tif_path in enumerate(tif_paths):
        img = read_data_from_tif(tif_path)
        img[img == -np.finfo(np.float32).max] = 0
        if config["normalize"] and not config["normalize_to_0_1"]:
            data[i, :, :] = normalize(img)
        elif not config["normalize"] and config["normalize_to_0_1"]:
            data[i, :, :] = normalize_to_0_1(img)
        elif config["normalize"] and config["normalize_to_0_1"]:
            raise ValueError("config['normalize'] 和 config['normalize_to_0_1'] 不能同时为 True")
        else:
            data[i, :, :] = img

    return data

def augment_data(data, target):
    # 4个角度进行翻转
    rotations = [0, 90, 180, 270]
    augmented_data = []
    augmented_target = []

    for x, y in zip(data, target):
        for angle in rotations:
            # 按照上面设定角度翻转
            rotated_image = rotate(x, angle, axes=(1,2), reshape=False)
            augmented_data.append(rotated_image)
            augmented_target.append(y)
            # 上下翻转
            flipped_image = np.fliplr(rotated_image)
            augmented_data.append(flipped_image)
            augmented_target.append(y)

    return np.array(augmented_data), np.array(augmented_target)

class creat_dataset():

    def __init__(self, tensor_data):
        self.data = tensor_data
        # 因子数
        self.F = tensor_data.shape[0]
        # 高
        self.h = tensor_data.shape[1]
        # 宽
        self.w = tensor_data.shape[2]
        self.all_results = None

    def creat_new_tensor(self):
        # 扩大图像边缘
        new_tensor = np.zeros((self.F, self.h + 2, self.w + 2))
        # 将数据值赋值到图像中心
        new_tensor[:, 1:self.h + 1, 1:self.w + 1] = self.data
        return new_tensor

    def get_images_labels(self, data, labels, mode='train'):
        train_images, train_labels = [], []
        valid_images, valid_labels = [], []
        count_0, count_1, count_2, count_3 = 0, 0, 0, 0
        if self.all_results is not None:
            train_images, train_labels, valid_images, valid_labels = self.all_results
            if mode == "train":
                print('训练集： ' + str(len(train_images)), str(len(train_labels)))
                return train_images, train_labels
            else:
                print('测试集： ' + str(len(valid_images)), str(len(valid_labels)))
                return valid_images, valid_labels
        # 遍历label
        for i in range(config["height"]):
            for j in range(config["width"]):
                # 训练集
                if labels[i, j] == 0 or labels[i, j] == 2:
                    # 读取 3 × 3 × 13 区域
                    train_images.append(data[:, i - 1:i + 2, j - 1:j + 2].astype(np.float32))
                    # 滑坡点
                    if labels[i, j] == 0:
                        count_0 += 1
                        train_labels.append(1)
                    # 非滑坡点
                    if labels[i, j] == 2:
                        count_2 += 1
                        train_labels.append(0)
                # 验证集
                if labels[i, j] == 1 or labels[i, j] == 3:
                    # 读取 3 × 3 × 13 区域
                    valid_images.append(data[:, i - 1:i + 2, j - 1:j + 2].astype(np.float32))
                    # 滑坡点
                    if labels[i, j] == 1:
                        count_1 += 1
                        valid_labels.append(1)
                    # 非滑坡点
                    if labels[i, j] == 3:
                        count_3 += 1
                        valid_labels.append(0)
        print("label 为 0，1，2，3的像素点个数分别为{},{},{},{}".format(count_0, count_1, count_2, count_3))
        # 数据增强
        if config["DataAugmentation"]:
            train_images, train_labels = augment_data(train_images, train_labels)
        # 按照 S 型排列将 3 × 3 × 13 转换为 9 × 13
        index_array = np.array([[0, 0], [0, 1], [0, 2], [1, 2], [1, 1], [1, 0], [2, 0], [2, 1], [2, 2]])
        train_images = [np.array([image[:, index_array[i, 0], index_array[i, 1]] for i in range(9)]) for image in train_images]
        valid_images = [np.array([image[:, index_array[i, 0], index_array[i, 1]] for i in range(9)]) for image in valid_images]
        if self.all_results is None:
            self.all_results = train_images, train_labels, valid_images, valid_labels
        if mode == "train":
            print('训练集： ' + str(len(train_images)), str(len(train_labels)))
            return train_images, train_labels
        else:
            print('测试集： ' + str(len(valid_images)), str(len(valid_labels)))
            return valid_images, valid_labels


def get_train_data(config, creat):
    # 返回扩大图像边缘的地图
    data = creat.creat_new_tensor() 
    # 读取label文件并转换为nparray
    labels = read_data_from_tif(config["label_path"])
    train_images, train_labels = creat.get_images_labels(data, labels, mode='train')
    return train_images, train_labels


def get_test_data(config, creat):
    # 返回扩大图像边缘的地图
    data = creat.creat_new_tensor() 
    # 读取label文件并转换为nparray
    labels = read_data_from_tif(config["label_path"])
    valid_images, valid_labels = creat.get_images_labels(data, labels, mode='valid')
    return valid_images, valid_labels


def shuffle_image_label_0(images, labels):
    """
    Randomly disrupt two list with the same shuffle
    """
    # 将两个列表（images和labels）随机打乱顺序，同时保持它们之间的对应关系
    # 这通常用于数据增强或者数据集准备阶段，以提高模型的泛化能力和训练效果。
    # 使用同一随机数种子进行打乱
    randnum = random.randint(0, len(images) - 1)
    random.seed(randnum)
    random.shuffle(images)
    random.seed(randnum)
    random.shuffle(labels)
    return images, labels


def train_data(creat):
    data_file = 'train_data{}.pkl'.format('_Aug' if config["DataAugmentation"] else'')
    # 判断缓存数据是否存在
    if os.path.exists(data_file):
        saved_config_dict = {}
        with open(data_file, 'rb') as f:
            # 读取缓存数据
            data = pickle.load(f)
        # 读取缓存数据中的配置表
        saved_config = data.get('config', {})
        for item in saved_config:
            key = item[0]
            value = item[1]
            saved_config_dict[key] = value
        keys = ['label_path', 'feature', 'width', 'height', 'normalize', 'normalize_to_0_1', "DataAugmentation"]
        # 获取当前配置表每项的值
        values = [config.get(key) for key in keys]
        # 获取缓存数据中的数据表每项的值
        saved_values = [saved_config_dict.get(key) for key in keys]
        # 判断当前配置表每项的值是否与缓存数据中的数据表每项的值相同
        if np.all(values == saved_values):
            # 如果相同，就直接读取缓存数据
            train_images = data['train_images']
            train_labels = data['train_labels']
        else:
            # 如果不相同，重新读取数据
            train_images, train_labels = get_train_data(config, creat)
            # 获取当前配置表
            config_list = [[k, v] for k, v in config.items()]
            data = {'train_images': train_images, 'train_labels': train_labels, 'config': config_list}
            # 将当前地图数据和配置表保存为缓存数据
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
    else:
        # 如果缓存数据不存在，读取数据
        train_images, train_labels = get_train_data(config, creat)
        config_list = [[k, v] for k, v in config.items()]
        data = {'train_images': train_images, 'train_labels': train_labels, 'config': config_list}
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)
    # 打乱数据，数据增强
    train_images, train_labels = shuffle_image_label_0(train_images, train_labels)
    return np.array(train_images).reshape((-1, 9, config["feature"])), np.array(
            train_labels).reshape((-1, 1))


def test_data(creat):
    data_file = 'valid_data.pkl'
    # 判断缓存数据是否存在
    if os.path.exists(data_file):
        saved_config_dict = {}
        with open(data_file, 'rb') as f:
            # 读取缓存数据
            data = pickle.load(f)
        # 读取缓存数据中的配置表
        saved_config = data.get('config', {})
        for item in saved_config:
            key = item[0]
            value = item[1]
            saved_config_dict[key] = value
        keys = ['label_path', 'feature', 'width', 'height', 'normalize', 'normalize_to_0_1']
        # 获取当前配置表每项的值
        values = [config.get(key) for key in keys]
        # 获取缓存数据中的数据表每项的值
        saved_values = [saved_config_dict.get(key) for key in keys]
        # 判断当前配置表每项的值是否与缓存数据中的数据表每项的值相同
        if np.all(values == saved_values):
            # 如果相同，就直接读取缓存数据
            valid_images = data['valid_images']
            valid_labels = data['valid_labels']
        else:
            # 如果不相同，重新读取数据
            valid_images, valid_labels = get_test_data(config, creat)
            # 获取当前配置表
            config_list = [[k, v] for k, v in config.items()]
            data = {'valid_images': valid_images, 'valid_labels': valid_labels, 'config': config_list}
            # 将当前地图数据和配置表保存为缓存数据
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
    else:
        # 如果缓存数据不存在，读取数据
        valid_images, valid_labels = get_test_data(config, creat)
        config_list = [[k, v] for k, v in config.items()]
        data = {'valid_images': valid_images, 'valid_labels': valid_labels, 'config': config_list}
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)
    # 验证集不进行 shuffle 数据增强
    return np.array(valid_images).reshape((-1, 9, config["feature"])), np.array(
            valid_labels).reshape((-1, 1))


def save_to_tif(pred_result, save_path):
    """
    :保存LSM
    """
    img = pred_result.reshape((config["height"], config["width"]))
    im_geotrans, im_prof = [], []
    for tif_path in config["data_path"]:  # 取仿射矩阵、投影坐标
        tif = gdal.Open(tif_path)
        im_geotrans.append(tif.GetGeoTransform())
        im_prof.append(tif.GetProjection())

    if 'int8' in img.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    im_height, im_width = img.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(save_path, im_width, im_height, 1, datatype)
    dataset.GetRasterBand(1).WriteArray(img)  # 写入数组数据
    dataset.SetGeoTransform(im_geotrans[0])  # 写入仿射变换参数
    dataset.SetProjection(im_prof[0])  # 写入投影
    del dataset
    print('ok')
