import matplotlib.pyplot as plt
from dataset import Dataset
class ImageDisplayer:
    def __init__(self):
        pass
    
    def show_singleimg(self,img):
        plt.imshow(img, cmap = 'gray')
        plt.title('Destination'), plt.axis('off')
        plt.show()
        
    def show_imgDataSet(self,img_data_set:Dataset):
        img = img_data_set.data
        plt.figure(figsize=(10, 10))
        for i in range(0,10):
            for j in range(0,10):
                plt.subplot(10,10,i*10 + j + 1)
                plt.imshow(img[i*10 + j], cmap = 'gray')
                plt.axis('off')
        plt.show()