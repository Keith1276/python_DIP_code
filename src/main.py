from dataset import Dataset
from feature_extraction import FeatureExtractor
from img_show import ImageDisplayer
from feature_match import FeatureMatch
import cv2
import numpy as np
import math
def main():
    
    #图像展示器
    img_displayer = ImageDisplayer()
    
    # 数据预处理部分
    dataset_path = 'asset/att_faces'
    orl_dataset = Dataset(dataset_path)
    # img_displayer.show_imgDataSet(orl_dataset)
    
    
    #特征提取
    feature_Extractor = FeatureExtractor()
    KL_base, flattened_img_lib = feature_Extractor.get_KL_feature(orl_dataset.data, rate = 1.0)
    
    #特征匹配
    feature_match = FeatureMatch(flattened_img_lib,orl_dataset.labels,KL_base)
    
    for order in range(0,400):
        test_img = orl_dataset.data[order]
        res = feature_match.match_new_img(test_img)
        print('匹配人物编号：',res,'正确人物编号', math.ceil(order / 10))
    
if __name__ == "__main__":
    main()