import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()):
    # torch.cuda.set_device(0)     
    device = torch.device('cuda') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                        )
        else:
            self.actor = nn.Sequential(
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
                            nn.Linear(64,action_dim),
                            nn.Softmax(dim=-1)
                            )            
            # self.actor = nn.Sequential(
            #                 nn.Conv2d(3, 64, kernel_size = 3, stride = 1),
            #                 nn.ReLU(),
            #                 nn.MaxPool2d(2,2),
            #                 nn.Conv2d(64,64, kernel_size = 3, stride = 1),
            #                 nn.ReLU(),
            #                 nn.MaxPool2d(2,2),
            #                 nn.Conv2d(64,64, kernel_size = 3, stride = 1),
            #                 nn.ReLU(),
            #                 nn.MaxPool2d(2,2),                                                        
            #                 nn.Flatten(),   
            #                 nn.Linear(6400,512),
            #                 nn.ReLU(),
            #                 nn.Linear(512, 512),
            #                 nn.ReLU(),
            #                 nn.Linear(512,action_dim),
            #                 nn.Softmax(dim=-1)
            #                 )                                        
        # critic
        self.critic = nn.Sequential(
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
                            nn.Linear(64,1)
                            )


        # self.critic = nn.Sequential(
        #                     nn.Conv2d(3, 64, kernel_size = 3, stride = 1),
        #                     nn.ReLU(),
        #                     nn.MaxPool2d(2,2),
        #                     nn.Conv2d(64,64, kernel_size = 3, stride = 1),
        #                     nn.ReLU(),
        #                     nn.MaxPool2d(2,2),
        #                     nn.Conv2d(64,64, kernel_size = 3, stride = 1),
        #                     nn.ReLU(),
        #                     nn.MaxPool2d(2,2),                                                        
        #                     nn.Flatten(),   
        #                     nn.Linear(6400,512),
        #                     nn.ReLU(),
        #                     nn.Linear(512, 512),
        #                     nn.ReLU(),
        #                     nn.Linear(512,64),
        #                     nn.ReLU(),
        #                     nn.Linear(64,1)
        #                     )