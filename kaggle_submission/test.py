
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from kaggle_environments import make, evaluate
import MCTS_sumbit as mcts



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


def my_agent1(observation, configuration):
    mark = observation.mark
    row = configuration.rows
    col = configuration.columns
    
    board = np.array(observation.board).reshape(row, col)
    
    
    action, _ = mcts.select_mcts_action(initial_board=board, initial_player_to_act=mark, config=configuration,
                       model=model1, n_simulations = 50 , c_puct = 0, c_fpu=0, device =device,
                       np_rng=np.random.default_rng(), dirichlet_alpha=0, dirichlet_epsilon=0,
                       temperature=0.0, log_debug=False)
    
        
    return action

def load_model(model, path, filename="connectx_model.pth", device=None):
    """
    Loads the model's state dictionary from the specified path.

    Args:
        model (nn.Module): The model instance to load parameters into.
        path (str): Directory containing the model file.
        filename (str): Name of the model file.
        device (torch.device, optional): Device to load the model onto (e.g., 'cuda', 'cpu').
                                         If None, loads to CPU by default unless map_location overrides.

    Returns:
        nn.Module: The model with loaded parameters.
    """
    full_path = os.path.join(path, filename)
    if not os.path.exists(full_path):
  
        raise FileNotFoundError(f"Model file not found: {full_path}")

    # <<< GPU/Device Change >>>
    # Determine map_location based on the desired device
    if device:
        map_location = device
    else:
        # Default map_location based on availability if device not specified
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        state_dict = torch.load(full_path, map_location=map_location)
        model.load_state_dict(state_dict)
        # Ensure the model itself is on the correct device after loading state_dict
        model.to(map_location)

        return model
    except Exception as e:

        raise


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1 = ConnectXNet()
    model1 = load_model(model1, '../models/checkpoints', 'model_iter_100.pth', device=device)

    model1.eval()
    env = make("connectx", debug=True)
    env.reset()
    env.run([my_agent1, "negamax"])
    print("Game finished.")
    print(env.render(mode="ipython"))