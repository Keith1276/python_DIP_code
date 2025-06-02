from dataset import Dataset
from feature_extraction import FeatureExtractor
from img_show import ImageDisplayer
from feature_match import FeatureMatch
import math
import torch
import torch.nn as nn
def main():
    
    #图像展示类
    img_displayer = ImageDisplayer()
    # 数据预处理部分，训练集 验证 测试 6:2:2
    train_len = 6
    test_len = 2
    total = train_len + test_len
    dataset_path = 'asset/att_faces'
    orl_train_dataset = Dataset(dataset_path,is_train=True,train_len=train_len,test_len=test_len)
    orl_test_dataset = Dataset(dataset_path,is_train=False,train_len=train_len,test_len=test_len)
    
    img_displayer.show_imgDataSet(orl_train_dataset,train_len)
    # img_displayer.show_imgDataSet(orl_test_dataset,10-train_len)
    
    #特征提取
    feature_Extractor = FeatureExtractor()
    
    #特征脸空间和样本 
    mean_data, KL_base, flattened_img_lib = feature_Extractor.get_KL_feature(orl_train_dataset.data, rate = 0.2)
    # img_displayer.show_mean_face(mean_data)
    # img_displayer.show_eigen_faces(KL_base)
    #特征匹配
    feature_match = FeatureMatch(mean_data,flattened_img_lib,orl_train_dataset.labels,KL_base)
    
    #测试
    accuracy = 0
    test_data_len = len(orl_test_dataset.data)
    
    for order in range(1,test_data_len+1):
        test_img = orl_test_dataset.data[order-1]
        #返回在前k个坐标上投影的重构图像和最佳匹配图像
        res,reconstruct_vector,best_match_vector= feature_match.match_new_img(test_img)
        if  res == math.ceil(order / (total - train_len)):
            accuracy += 1/test_data_len
        else:
            # img_displayer.show_mismatchs(reconstruct_vector,best_match_vector,mean_data, KL_base, math.ceil(order / (total - train_len)),res)
            pass
        print('匹配人物编号：',res,'正确人物编号', math.ceil(order / (total - train_len)))
    
    print('预测准确率为：',accuracy)
if __name__ == "__main__":
    main()