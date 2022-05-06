
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, action_dim):
        super(Model, self).__init__()

        self.net = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 3, stride = 1),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(64,64, kernel_size = 3, stride = 1),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),
                        nn.Flatten(),   
                        nn.Linear(33856,512),
                        nn.ReLU(),
                        nn.Linear(512, 64),
                        nn.ReLU(),
                        nn.Linear(64,action_dim)
                    )

    def forward(self, x):
        return self.net(x)