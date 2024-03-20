import os
import re
from osgeo import gdal

"""
        1.以下代码每次调用config均会执行更新，以便读取data相关参数
        2.可修改newdata_path使用自己数据集，其余数据相关参数会自动更新
        3.所有data和label尺寸大小均需完全一致
"""

config = {
    "newdata_path": "./data/",  # 可修改存放数据的根目录，使用自己数据
    "data_path": ["origin_data/aspect_10.tif",
                  "origin_data/duanceng_5.tif",
                  "origin_data/elevation_5.tif",
                  "origin_data/gcyz_3.tif",
                  "origin_data/gougumd_5.tif",
                  "origin_data/qifudu_5.tif",
                  "origin_data/river_5.tif",
                  "origin_data/road_5.tif",
                  "origin_data/slope_5.tif",
                  "origin_data/slope_5.tif",
                  ],
    "label_path": "origin_data/label1.tif",
    # 标签TIF文件  需包括 0(训练集滑坡) 1(测试集滑坡) 2(训练集非滑坡) 3(测试集非滑坡)   0+1=2+3 且 (0+2):(1+3)=7:3 or 8:2
    "feature": 13,     # 因子数
    "width": 3368,     # 宽度
    "height": 2626,    # 高度
    "batch_size": 16392,  # 训练batch_size大小 
    "epochs": 200,  # 训练epoch数量
    "Cutting_window": 64,  # 切片大小，用于预测画图
    "device": "cuda:0",    # 可以为"cuda"或"cpu"，"cuda:0"=指定第一张GPU
    "lr": 0.001,         # 学习率
    "normalize": False,  # 标准化
    "normalize_to_0_1": True,  # 归一化
    "DataAugmentation": True   # 数据增强
}

data_path = []
for tif_data in os.listdir(config["newdata_path"]):
    if tif_data.endswith('tif'):
        if re.match('label', tif_data):
            config["label_path"] = config["newdata_path"] + tif_data  # 更新标签路径
            continue
        temp = config["newdata_path"] + tif_data
        data_path.append(temp)
        
###   更新config字典    ###
config["data_path"] = data_path
config["feature"] = len(data_path)
tif = gdal.Open(config["data_path"][0])
config["width"], config["height"] = tif.RasterXSize, tif.RasterYSize
