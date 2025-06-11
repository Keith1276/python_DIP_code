import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from loss import ContrastiveLoss
from dataset import SiameseNetworkDataset
from model import leNet

#####################################超参数定义################################
train_batch_size = 8
train_number_epochs = 100
def imshow(img ,text = None,should_save = False, path = None):
    #展示一幅tensor图像
    npimg = img.numpy()
    plt.axis('off')
    if text:
        plt.text(75, 8, str(text), style='italic',
        bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    if path:
        plt.savefig(path)
    plt.show()

def show_valid_plot(iteration, acc, path):
    plt.plot(iteration, acc)
    plt.ylabel('accuracy')
    plt.xlabel('batch')
    if path:
        plt.savefig(path)
def show_train_plot(iteration, loss, path):
    plt.plot(iteration, loss)
    plt.ylabel('loss')
    plt.xlabel('batch')
    if path:
        plt.savefig(path)
    plt.show()
        

################################  获得train dataset ################################
training_dir = "../asset/orl_faces/train"
folder_dataset = torchvision.datasets.ImageFolder(root = training_dir)


#这里这么高质量的图像视觉上用不着高通滤波和高斯噪声
transform = transforms.Compose([transforms.Resize((100,100)),
                                transforms.ToTensor(),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.Normalize((0.4515), (0.1978)),
                                transforms.GaussianBlur(3),
                                ])

#定义孪生数据集
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform = transform,
                                        should_invert=False,
                                        pos_rate = 0.5)

train_dataloader = DataLoader(siamese_dataset,
                              shuffle=True,
                              batch_size=train_batch_size)

    
device = torch.device("mps")
net = leNet().to(device)

criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005, weight_decay=1e-5) 

######################################### 训练开始 #################################################
counter = []
loss_history = []
iteration_number = 0
batch_num = len(siamese_dataset)/train_batch_size

for epoch in range(0, train_number_epochs):
    for i, data in enumerate(train_dataloader, 0):
        epoch_loss = 0.0
        img0, img1 , label = data
        #img0维度为torch.Size([32, 1, 100, 100])，32是batch，label为torch.Size([32, 1])
        img0, img1 , label = img0.to(device), img1.to(device), label.to(device) #数据移至GPU
        optimizer.zero_grad()
        output1,output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0 :
            iteration_number +=10
            counter.append(iteration_number)
            
    loss_history.append(loss_contrastive.item())
    print("Epoch number: {} , Current loss: {:.4f}\n".format(epoch,loss_contrastive.item()))
    
show_train_plot(counter, loss_history, '../output/loss.jpg')

model_path = '../model/lenet.pt'
torch.save(net, model_path)