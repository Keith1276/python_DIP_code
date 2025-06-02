import matplotlib.pyplot as plt
from dataset import Dataset
import numpy as np
class ImageDisplayer:
    def __init__(self):
        pass
    
    def show_singleimg(self,img):
        plt.imshow(img, cmap = 'gray')
        plt.title('Destination'), plt.axis('off')
        plt.show()
    def show_mean_face(self, mean_face):
        plt.figure(figsize=(5, 5))
        tmp = mean_face.reshape(112, 92)
        plt.imshow(tmp, cmap = 'gray')
        plt.title(f"Mean Face")
        plt.axis("off")
        plt.show()
    def show_eigen_faces(self, KL_base):
        # 假设每个特征向量可以 reshape 成 112x92 的图像
        num_eigenfaces = min(20, KL_base.shape[1])  # 取前20个或更少可用的特征脸
        plt.figure(figsize=(10, 5))

        for i in range(num_eigenfaces):
            eigenface = KL_base[:, i].reshape(112, 92)  # 根据实际图像大小调整 reshape 参数
            plt.subplot(4, 5, i + 1)  # 4行5列展示20个特征脸
            plt.imshow(eigenface, cmap='gray')
            plt.title(f"Eigenface {i+1}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()
    def show_best_matchs(self,img):
        pass
    
    def show_mismatchs(self, origin_vector, match_vector, mean_vector, KL_base,true_lable,false_lable):
        # 确保 origin_vector 和 match_vector 是一维数组 (k,)
        origin_vector = np.ravel(origin_vector)
        match_vector = np.ravel(match_vector)

        # 重建图像：将特征向量与 KL 基相乘并加回均值脸
        origin_img = np.dot(KL_base, origin_vector) + mean_vector
        match_img = np.dot(KL_base, match_vector) + mean_vector

        # Reshape 成图像格式
        origin_img = origin_img.reshape(112, 92)
        match_img = match_img.reshape(112, 92)

        # 显示图像
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(origin_img, cmap='gray')
        plt.title(true_lable)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(match_img, cmap='gray')
        plt.title(false_lable)
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    def show_imgDataSet(self,img_data_set:Dataset, kind_size):
        img = img_data_set.data
        plt.figure(figsize=(10, 10))
        for i in range(0,10):
            for j in range(0,kind_size):
                plt.subplot(10,kind_size,i * kind_size + j + 1)
                plt.imshow(img[i * kind_size + j], cmap = 'gray')
                plt.axis('off')
        plt.show()