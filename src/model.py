import torch.nn as nn

class leNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
        # 1 * 100 * 100
        nn.Conv2d(1, 16, kernel_size=7, padding=1),
        nn.ReLU(inplace=True),
        # nn.BatchNorm2d(6),
        nn.MaxPool2d((2,2)),
        # 6 * 48 * 48
        nn.Conv2d(16, 32, kernel_size=5),
        nn.ReLU(inplace=True),
        # nn.BatchNorm2d(16),
        nn.MaxPool2d((2,2)),
        # 16 * 22 * 22
        nn.Conv2d(32, 32, kernel_size=3),
        nn.ReLU(inplace=True),
        # nn.BatchNorm2d(32),
        nn.MaxPool2d((2,2)),
        # 32 * 10 * 10
        )
        self.fc = nn.Sequential(
            nn.Linear(32*10*10, 20),
            # nn.ReLU(inplace = True),
            # nn.Linear(128, 20)
        )
    def forward_once(self,x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
    def forward(self,input1,input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1,output2