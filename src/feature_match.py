from dataset import Dataset
from feature_extraction import FeatureExtractor
import numpy as np
from collections import Counter
class FeatureMatch:
    def __init__(self,flattened_img_lib,img_lables,KL_base):
        #构建匹配库
        self.KL_base = KL_base
        self.match_lib = self.construct_match_lib(flattened_img_lib, img_lables, KL_base)
    
    #需要构造一个匹配库
    def construct_match_lib(self, flattened_img_lib,img_lables,KL_base):
        feature_img_lib = np.dot(flattened_img_lib, KL_base)
        match_lib = {}
        for vector, label in zip(feature_img_lib, img_lables):
            key = tuple(vector)  # 将特征向量转为元组作为字典的键
            match_lib[key] = label
        return match_lib
    #对于新的灰度图像，knn匹配:
    def match_new_img(self, new_img,k = 3):
        # 提取新图像的特征向量
        new_img_feature = np.dot(new_img.reshape(1,10304), self.KL_base)

        # 找出匹配库中距离最近的三个样本
        top_k_labels = self._find_top_k_matches(new_img_feature, k=3)

        # 使用投票法决定最终 label
        predicted_label = Counter(top_k_labels).most_common(1)[0][0]
        return predicted_label

    def _find_top_k_matches(self, query_vector, k=3):
    # 计算所有样本与 query 的 L2 距离
        distances = []
        for vec_key, label in self.match_lib.items():
            vec = np.array(vec_key)
            dist = np.linalg.norm(query_vector - vec)
            distances.append((dist, label))

        # 按距离排序并取前 k 个
        distances.sort(key=lambda x: x[0])
        top_k_labels = [label for _, label in distances[:k]]

        return top_k_labels