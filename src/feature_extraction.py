import numpy as np
from dataset import Dataset
class FeatureExtractor:
    # 输入单张RGB图提取特征向量
    def __init__(self):
        pass
    #data为数据集里边的图像数据，不带标签，对整个数据集进行分析，返回最终选择的正交基即可
    #rate为svd加速后的基保留百分比
    def get_KL_feature(self, data, rate = 1.0):
        #数据中心化，之后计算小维度的特征值和特征向量即可
        flatten_data = self.img_flatten(data)
        #拿到前k个特征值对应的特征向量
        mean_data,KL_base = self.get_KL_base(flatten_data,rate)
        
        return mean_data, KL_base, flatten_data
        
    
    #利用SVD对KL变换进行加速，只要计算400开头的那个维度所包含的
    def get_KL_base(self, flatten_data, rate):
         # 中心化
        centered_data = flatten_data - np.mean(flatten_data, axis=0)
        
        # 使用SVD加速，直接分解中心化后的数据
        _, _, Vt = np.linalg.svd(centered_data, full_matrices=False)
        
        # 计算需要保留的主成分数量
        n_samples, n_features = centered_data.shape
        num = int(rate * min(n_samples, n_features))
        
        # 返回前num个主成分方向（转置为列向量）
        return np.mean(flatten_data, axis=0),Vt[:num, :].T
        

    def img_flatten(self, imgdata):
        # 将灰度图像转换为二维 numpy 数组 (样本数 x 像素数)
        flattened_data = np.array([img.flatten() for img in imgdata])  # shape: (num_samples, num_pixels)
        return flattened_data
