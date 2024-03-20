import read_data
import config
import generate_LSM
import train_cnn
from read_data import get_feature_data, creat_dataset
#实例化config
config = config.config

# True为开启训练
is_train = 0
# True为开启画图
is_saveLSM = 1

if __name__ == '__main__':
    if is_train:
        print('***************************************读取训练集 测试集***************************************')
        tensor_data = get_feature_data()  # 读取训练数据 feature*width*height
        creat = creat_dataset(tensor_data) # 创建地图
        all_data_train, all_target_train = read_data.train_data(creat) # 训练数据和标签
        all_data_val, all_target_val = read_data.test_data(creat) # 验证数据和标签
        print('*******************************************开始训练*******************************************')
        train_cnn.train(all_data_train, all_target_train, all_data_val, all_target_val)
    if is_saveLSM:
        generate_LSM.save_LSM()