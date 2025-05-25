from dataset import Dataset
from feature_extraction import FeatureExtractor
from img_show import ImageDisplayer
from feature_match import FeatureMatch
import cv2
import numpy as np
import math
def main():
    
    #图像展示类
    img_displayer = ImageDisplayer()
    # 数据预处理部分
    train_len = 7
    dataset_path = 'asset/att_faces'
    orl_train_dataset = Dataset(dataset_path,is_train=True,train_len=train_len)
    orl_test_dataset = Dataset(dataset_path,is_train=False,train_len=train_len)
    # img_displayer.show_imgDataSet(orl_dataset)
    # img_displayer.show_imgDataSet(orl_train_dataset)
    # img_displayer.show_imgDataSet(orl_test_dataset,10-train_len)
    
    #特征提取
    feature_Extractor = FeatureExtractor()
    
    #特征脸空间和样本 
    mean_data,KL_base, flattened_img_lib = feature_Extractor.get_KL_feature(orl_train_dataset.data, rate = 1.0)
    
    #特征匹配
    feature_match = FeatureMatch(mean_data,flattened_img_lib,orl_train_dataset.labels,KL_base)
    
    #测试
    accuracy = 0
    test_len = len(orl_test_dataset.data)

    for order in range(1,test_len+1):
        test_img = orl_test_dataset.data[order-1]
        res = feature_match.match_new_img(test_img)
        if  res == math.ceil(order / (10 - train_len)):
            accuracy += 1/test_len
        print('匹配人物编号：',res,'正确人物编号', math.ceil(order / (10 - train_len)))
    
    print('预测准确率为：',accuracy)
if __name__ == "__main__":
    main()