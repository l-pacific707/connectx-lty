# ConnectX AlphaZero-style Neural Network Architecture with Preprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchview import draw_graph
import logging
import logger_setup
import os

logger = logging.getLogger("ConnectXNN")
file_handler = logging.FileHandler('ConnectXNN.log')
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)




# === Residual Block ===
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

# === ConnectX Network ===
class ConnectXNet(nn.Module):
    def __init__(self, input_channels=3, board_size=(6, 7), n_actions=7):
        super().__init__()
        self.board_h, self.board_w = board_size

        # Shared conv body
        self.conv_in = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Residual blocks
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * self.board_h * self.board_w, n_actions)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.board_h * self.board_w, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res1(x)
        x = self.res2(x)
        p = self.policy_head(x)  # shape: (batch_size, 7)
        v = self.value_head(x)   # shape: (batch_size, 1)
        return p, v

# === Observation Preprocessing ===
def preprocess_input(env):
    """

    Args:
        env (connectX environment): attributes : {done, state, step, configuration}

    Returns:
        torch.tensor: (1,3,6,7)
    """
    board = np.array(env.state[0]["observation"]["board"]).reshape(6, 7) # 2d numnpy ndarray
    if env.state[0]["status"] == "ACTIVE":
        mark = 1
    elif env.state[1]["status"] == "ACTIVE":
        mark = 2
    else:
        logger.warning("undealt case. Consider there was logic error. Detail: both player is not ACTIVE.")

    P1 = (board == 1).astype(np.float32)
    P2 = ((board == 2)).astype(np.float32)
    if mark == 1:
        player_plane = np.full((6, 7), 1, dtype=np.float32)  
    elif mark == 2:
        player_plane = np.full((6, 7), -1, dtype=np.float32)  

    stacked = np.stack([P1, P2, player_plane])  # (3, 6, 7)
    return torch.tensor(stacked).unsqueeze(0)  # (1, 3, 6, 7)

def get_valid_actions(env):
    """Specially designed for ConnectX env.

    Args:
        env (connectx environment): _description_
    """
    board_width = env.configuration.columns
    board_height = env.configuration.rows
    board = np.array(env.state[0]["observation"]["board"]).reshape((board_height, board_width))
 
    valid_actions = [c for c in range(board_width) if board[0][c] == 0]
    return valid_actions

def select_action(model, env):
    input = preprocess_input(env)
    p_logits, v = model(input)
    p = torch.softmax(p_logits, dim=-1).detach().cpu().numpy().flatten()
    valid_actions = get_valid_actions(env)
    prob = np.zeros(env.configuration.columns)
    prob[valid_actions] = p[valid_actions]
    action = np.random.choice(env.configuration.columns, p=prob)
    return action, prob

def save_model(model, path, filename="connectx_model.pth"):
    """
    모델의 파라미터를 지정된 경로에 저장합니다.
    
    Args:
        model (nn.Module): 저장할 모델
        path (str): 저장할 디렉토리 경로
        filename (str): 저장할 파일명 (기본값: connectx_model.pth)
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    torch.save(model.state_dict(), full_path)
    print(f"model is saved in : {full_path}")

def load_model(model, path, filename="connectx_model.pth", map_location=None):
    """
    저장된 모델 파라미터를 로드합니다.

    Args:
        model (nn.Module): 파라미터를 로드할 모델 인스턴스
        path (str): 저장된 파일의 디렉토리
        filename (str): 파일명
        map_location: CPU에서 로드할 경우 "cpu", GPU면 None
    
    Returns:
        model (nn.Module): 로드된 파라미터를 가진 모델
    """
    full_path = os.path.join(path, filename)
    state_dict = torch.load(full_path, map_location=map_location)
    model.load_state_dict(state_dict)
    print(f"Model is loaded in : {full_path}")
    return model

    
    
# === Example Usage ===
if __name__ == "__main__":
    model = ConnectXNet()
    viz = draw_graph(model, input_size=(1, 3, 6, 7), expand_nested=True)
    viz.visual_graph.render("connectx_torchview", format="png")  # PNG 저장
    # dummy observation for testing
    from kaggle_environments import make
    env = make("connectx", debug=True)
    x = preprocess_input(env)
    p_logits, v = model(x)
    print("p_logits shape:", p_logits.shape)  # (1, 7)
    print("v shape:", v.shape)                # (1, 1)
