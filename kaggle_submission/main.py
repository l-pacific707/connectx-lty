import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

import lzma
import pickle
import sys



if os.path.exists('/kaggle/input/connectx-source-code-and-parameter/'):
    cwd = '/kaggle/input/connectx-source-code-and-parameter/'
elif os.path.exists('/kaggle_simulations/agent/'):
    cwd = '/kaggle_simulations/agent/'
else:
    cwd = './'  

sys.path.append(cwd) 


import MCTS_submit as mcts


def load_model():
    global model
    model = ConnectXNet()
    
    with lzma.open(os.path.join(cwd, "mydata.pkl.xz"), "rb") as f:
        import io
        buffer = f.read()
        state_dict = torch.load(io.BytesIO(buffer), map_location=torch.device("cpu"))
    
    model.load_state_dict(state_dict)
    model.eval()


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False) # Bias false if using BN
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False) # Bias false if using BN
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual # Add residual connection *before* final activation
        return F.relu(out)

class ConnectXNet(nn.Module):
    def __init__(self, input_channels=3, board_size=(6, 7), n_actions=7, num_res_blocks=5):
        super().__init__()
        self.board_h, self.board_w = board_size
        self.n_actions = n_actions

        # Shared conv body
        self.conv_in = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.board_h * self.board_w, n_actions)

        # Value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * self.board_h * self.board_w, 64)
        self.value_fc2 = nn.Linear(64, 1)


    def forward(self, x):
        # Ensure input is float
        x = x.float()
        # Shared body
        x = self.conv_in(x)
        x = self.res_blocks(x)

        # Policy head
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = torch.flatten(p, start_dim=1) # Flatten all dims except batch
        p = self.policy_fc(p) # Output logits, shape: (batch_size, n_actions)

        # Value head
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)
        v = torch.flatten(v, start_dim=1) # Flatten all dims except batch
        v = self.value_fc1(v)
        v = F.relu(v)
        v = self.value_fc2(v)
        v = torch.tanh(v) # Output value in [-1, 1], shape: (batch_size, 1)

        return p, v


model = None
device = torch.device("cpu")
load_model()
np_rng = np.random.default_rng()

def act(observation, configuration):
    global model, device
    if model is None:
        load_model()
    mark = observation.mark
    row = configuration.rows
    col = configuration.columns
    
    board = np.array(observation.board).reshape(row, col)
    global np_rng
    
    action, _ = mcts.select_mcts_action(initial_board=board, initial_player_to_act=mark, config=configuration,
                       model=model, n_simulations = 90 , c_puct = 1.0, c_fpu=0, device =device,
                       np_rng=np_rng, dirichlet_alpha=0, dirichlet_epsilon=0,
                       temperature=0.0, log_debug=False)
    
        
    return action
