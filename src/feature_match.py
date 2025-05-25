import numpy as np
class FeatureMatch:
    def __init__(self,mean_data,flattened_img_lib,img_lables,KL_base):
        #构建匹配库
        self.avg_face = mean_data
        self.KL_base = KL_base
        self.match_lib = self.construct_match_lib(flattened_img_lib, img_lables, KL_base)
    
    #需要构造一个匹配库
    def construct_match_lib(self, flattened_img_lib,img_lables,KL_base):
        
        #图像库先减去平均脸
        flattened_img_lib = flattened_img_lib - self.avg_face
            
        feature_img_lib = np.dot(flattened_img_lib, KL_base)
        
        match_lib = {}
        for vector, label in zip(feature_img_lib, img_lables):
            key = tuple(vector)  # 将特征向量转为元组作为字典的键
            match_lib[key] = label
        return match_lib
    
    
    #对于新的灰度图像，knn匹配:
    def match_new_img(self, new_img,k = 1):
        # 提取新图像的特征向量
        new_img = new_img.flatten() - self.avg_face
        # new_img = new_img.reshape(1,10304)
        new_img_feature = np.dot(new_img, self.KL_base)

        # 找出匹配库中距离最近的三个样本
        top_k_labels = self._find_top_k_matches(new_img_feature, k)

        # 使用投票法决定最终 label
        label_counts = {}
        for label in top_k_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        # 找出最高票数的标签（如果有多个相同票数，则取第一个出现的标签）
        max_count = -1
        predicted_label = None
        for label in top_k_labels:
            count = label_counts[label]
            if count > max_count:
                max_count = count
                predicted_label = label
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