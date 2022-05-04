
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, action_dim, is_actor):
        super(Model, self).__init__()

        self.is_actor = is_actor
        self.action_dim = action_dim
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
                        nn.ReLU()    
                    )

    def forward(self, x):
        shared = self.net(x)

        # run actor network
        if self.is_actor == True:
            last_layer = nn.Linear(64,self.action_dim)
            soft_activate = nn.Softmax(dim=-1)
            return soft_activate(last_layer(shared))
        # run critic network
        else:
            last_layer = nn.Linear(64,1)
            return last_layer(shared)