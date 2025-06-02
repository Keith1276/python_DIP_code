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
    def match_new_img(self, new_img, k=1):
        # 提取新图像的特征向量
        new_img = new_img.flatten() - self.avg_face
        new_img_feature = np.dot(new_img, self.KL_base)

        # 找出匹配库中距离最近的 k 个样本及其距离
        top_k_matches = self._find_top_k_matches_with_vectors(new_img_feature, k)

        # 使用投票法决定最终 label
        label_counts = {}
        for _, label in top_k_matches:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        # 找出最高票数的标签（如果有多个相同票数，则取第一个出现的标签）
        max_count = -1
        predicted_label = None
        for _, label in top_k_matches:
            count = label_counts[label]
            if count > max_count:
                max_count = count
                predicted_label = label

        # 返回预测标签和最匹配的特征向量（即 top1）
        return predicted_label,new_img_feature,top_k_matches[0][0]

    def _find_top_k_matches_with_vectors(self, query_vector, k=3):
    # 计算所有样本与 query 的 L2 距离，并保留原始向量
        distances = []
        
        use_cosine_distance = True
        if(use_cosine_distance):
            for vec_key, label in self.match_lib.items():
                vec = np.array(vec_key)
                # 计算改进的余弦相似度（例如：1 - 余弦相似度作为“距离”）
                cosine_similarity = np.dot(query_vector, vec) / (np.linalg.norm(query_vector) * np.linalg.norm(vec))
                modified_cosine_distance = 1 - cosine_similarity
                distances.append((vec, modified_cosine_distance, label))  # 存储向量、改进的余弦距离、标签
        else:
            for vec_key, label in self.match_lib.items():
                vec = np.array(vec_key)
                dist = np.linalg.norm(query_vector - vec)
                distances.append((vec, dist, label))  # 存储向量、距离、标签

        # 按距离排序并取前 k 个
        distances.sort(key=lambda x: x[1])  # 按距离排序
        top_k_matches = [(item[0], item[2]) for item in distances[:k]]  # (特征向量, 标签)

        return top_k_matches