from torch import nn
import numpy as np

class Model(nn.Module):
    def __init__(self, output_size, output_std):
        super(Model,self).__init__()
        self.net = nn.Sequential(self.layer_init(
            nn.Conv2d(3,32,kernel_size=5,stride=1,padding=2)),#240,640
            nn.Tanh(),
            nn.MaxPool2d(2,2),
            self.layer_init(nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2)),#120,320
            nn.Tanh(),
            nn.MaxPool2d(4,4),
            self.layer_init(nn.Conv2d(64,128,kernel_size=3,padding=1)),#30,80
            nn.Tanh(),
            nn.MaxPool2d(3,3),
            self.layer_init(nn.Conv2d(128,128,kernel_size=3,padding=1)),#10,26
            nn.Tanh(),
            nn.Flatten(),
            self.layer_init(nn.Linear(33280, 1024)),
            nn.Tanh(),
            self.layer_init(nn.Linear(1024, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, output_size), std=output_std)
        )

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

    