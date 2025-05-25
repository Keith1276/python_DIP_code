import matplotlib.pyplot as plt
from dataset import Dataset
class ImageDisplayer:
    def __init__(self):
        pass
    
    def show_singleimg(self,img):
        plt.imshow(img, cmap = 'gray')
        plt.title('Destination'), plt.axis('off')
        plt.show()
    
    def show_best_matchs(self,img):
        pass
    def show_imgDataSet(self,img_data_set:Dataset, kind_size):
        img = img_data_set.data
        plt.figure(figsize=(10, 10))
        for i in range(0,10):
            for j in range(0,kind_size):
                plt.subplot(10,kind_size,i * kind_size + j + 1)
                plt.imshow(img[i * kind_size + j], cmap = 'gray')
                plt.axis('off')
        plt.show()