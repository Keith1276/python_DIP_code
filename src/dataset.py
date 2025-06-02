import cv2
import os
import numpy as np

class Dataset:
    def __init__(self, path,is_train=True,train_len=7,test_len = 3,apply_clahe=False):
        self.path = path
        self.data, self.labels= self.load_orl_dataset(path, is_train,train_len,test_len)
        if apply_clahe:
            self.img_clahe()
    def load_orl_dataset(self, path, is_train=True,train_len=7,test_len =3):
        images = []
        labels = []
        
        for person_id in range(1, 41):  
            person_dir = os.path.join(path, f's{person_id}')
            if is_train:
                for image_id in range(1, train_len+1): 
                    img_path = os.path.join(person_dir, f'{image_id}.pgm')
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images.append(img)
                        labels.append(person_id)
            else:
                for image_id in range(train_len+1, train_len+1+test_len): 
                    img_path = os.path.join(person_dir, f'{image_id}.pgm')
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images.append(img)
                        labels.append(person_id)
                     
        return np.array(images), np.array(labels)
    
    #灰度直方图均衡化
    def img_clahe(self):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_images = []
        for img in self.data:
            enhanced_img = clahe.apply(img)
            enhanced_images.append(enhanced_img)
        self.data = np.array(enhanced_images)  # 更新数据集为增强后的图像
    #最初使用EVD方法实在太难算，准备降维分块做，但是后来发现可以SVD加速，下边的下采样也就失效了
    def img_down_sample(self,dest_width =46, dest_height = 56):
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                self.data[i * 10 + j] = cv2.resize(self.data[i * 10 + j], (dest_width, dest_height))
    
    