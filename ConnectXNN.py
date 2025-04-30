import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchview import draw_graph
from logger_setup import get_logger # Assuming you have this setup
import os


logger = get_logger("ConnectXNN","ConnectXNN.log")
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

# === ConnectX Network ===
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

# === Observation Preprocessing ===
def preprocess_input(env):
    """
    Preprocesses the ConnectX environment state into a format suitable for the NN.
    Input: Kaggle ConnectX environment object.
    Output: torch.Tensor of shape (1, 3, board_height, board_width).
            Channels: (Player 1 stones, Player 2 stones, Player-to-move indicator)
    """
    if env is None or env.state is None or not env.state:
         logger.error("Invalid environment provided to preprocess_input.")
         # Return a dummy tensor or raise error
         return torch.zeros((1, 3, 6, 7)) # Assuming default 6x7 board

    board_width = env.configuration.columns
    board_height = env.configuration.rows
    # Board is a 1D list in env.state[0]['observation']['board']
    board = np.array(env.state[0]["observation"]["board"], dtype=np.float32).reshape((board_height, board_width))

    # Determine whose turn it is (mark)
    if env.state[0]["status"] == "ACTIVE":
        mark = 1 # Player 1's turn
    elif env.state[1]["status"] == "ACTIVE":
        mark = 2 # Player 2's turn
    else: # Game is DONE or in error state
        logger.debug("Preprocessing input for a non-active game state.")
        # The 'player to move' plane might not be meaningful here, but we need to provide something.
        # Let's default to player 1's perspective or determine based on piece count.
        p1_pieces = np.sum(board == 1)
        p2_pieces = np.sum(board == 2)
        mark = 1 if p1_pieces <= p2_pieces else 2 # Guess based on who likely moved last

    # Create planes
    P1_plane = (board == 1).astype(np.float32)
    P2_plane = (board == 2).astype(np.float32)

    # Player-to-move plane: 1 if current player is P1, -1 if current player is P2
    # Renamed from 'player_plane' for clarity
    turn_plane = np.ones((board_height, board_width), dtype=np.float32) if mark == 1 else \
                 np.full((board_height, board_width), -1.0, dtype=np.float32)

    # Stack planes: (Player 1, Player 2, Turn Indicator)
    stacked = np.stack([P1_plane, P2_plane, turn_plane]) # Shape: (3, H, W)

    # Add batch dimension and convert to tensor
    # The tensor should be created on CPU first, moving to GPU happens later if needed.
    return torch.from_numpy(stacked).unsqueeze(0) # Shape: (1, 3, H, W)


def get_valid_actions(env):
    """Get a list of valid actions (columns where a piece can be dropped)."""
    board = np.array(env.state[0]["observation"]["board"])
    columns = env.configuration.columns
    # Valid if the top cell (index c for 1D board) is empty (0)
    return [c for c in range(columns) if board[c] == 0]

def select_action_from_net(model, env, device):
    """
    Selects an action directly using the network's policy head (no MCTS).
    Used for faster evaluation games.
    Returns action index and policy probabilities.
    """
    model.eval() # Ensure model is in eval mode
    with torch.no_grad():
        input_tensor = preprocess_input(env).to(device) # <<< Move input to device
        p_logits, _ = model(input_tensor)
        p_logits = p_logits.squeeze(0) # Remove batch dim

    valid_actions = get_valid_actions(env)
    if not valid_actions:
        logger.warning("select_action_from_net called with no valid actions.")
        return None, np.zeros(env.configuration.columns) # No action possible

    # Mask invalid actions in logits before softmax
    mask = torch.full_like(p_logits, -float('inf'))
    mask[valid_actions] = p_logits[valid_actions]

    # Apply softmax to get probabilities
    probabilities = F.softmax(mask, dim=0).cpu().numpy() # Move to CPU for numpy

    # Normalize probabilities just in case (should sum to 1 already)
    prob_sum = np.sum(probabilities)
    if prob_sum > 1e-6:
         probabilities = probabilities / prob_sum
    else:
         # If all probabilities are zero (e.g., network output issues), use uniform valid
         logger.warning("Network output probabilities sum to zero. Using uniform valid.")
         probabilities = np.zeros_like(probabilities)
         prob = 1.0 / len(valid_actions)
         for action in valid_actions:
              probabilities[action] = prob


    # Choose action based on probabilities
    try:
         action = np.random.choice(len(probabilities), p=probabilities)
    except ValueError as e:
         logger.error(f"Error choosing action in select_action_from_net. Probs: {probabilities}. Error: {e}")
         # Fallback to uniform valid action
         action = np.random.choice(valid_actions)

    return action, probabilities

def save_model(model, path, filename="connectx_model.pth"):
    """
    Saves the model's state dictionary to the specified path.
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    try:
        # Save the state_dict (device-agnostic)
        torch.save(model.state_dict(), full_path)
        logger.info(f"Model state_dict saved to: {full_path}")
    except Exception as e:
        logger.error(f"Failed to save model to {full_path}: {e}")


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
        logger.error(f"Model file not found at: {full_path}")
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
        logger.info(f"Model state_dict loaded from: {full_path} onto device {map_location}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {full_path}: {e}")
        raise


# === Example Usage ===
if __name__ == "__main__":
    # Setup basic logging if logger_setup is not available
    try:
        from logger_setup import get_logger
    except ImportError:
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("ConnectXNN_Test")

    # <<< GPU Change Start >>>
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info("CUDA device found, using GPU.")
    else:
        dev = torch.device("cpu")
        logger.info("CUDA device not found, using CPU.")
    # <<< GPU Change End >>>

    model = ConnectXNet().to(dev) # <<< Create model and move to device

    # Optional: Print model summary or visualize
    print(model)
    try:
        # Visualize the graph (requires torchview and graphviz)
        graph = draw_graph(model, input_size=(1, 3, 6, 7), expand_nested=True, device=dev)
        graph.visual_graph.render("connectx_network_torchview", format="png")
        logger.info("Network graph saved to connectx_network_torchview.png")
    except ImportError:
        logger.warning("torchview not installed, skipping graph visualization.")
    except Exception as e:
        logger.warning(f"torchview visualization failed: {e}")


    # Test preprocessing and forward pass
    from kaggle_environments import make
    env = make("connectx", debug=True)
    env.reset()

    try:
        # Preprocess
        input_tensor = preprocess_input(env)
        logger.info(f"Input tensor shape: {input_tensor.shape}") # Should be [1, 3, 6, 7]

        # Move input tensor to the model's device
        input_tensor = input_tensor.to(dev)

        # Forward pass
        model.eval() # Set to evaluation mode
        with torch.no_grad():
            p_logits, v = model(input_tensor)

        logger.info(f"Policy logits shape: {p_logits.shape}") # Should be [1, 7]
        logger.info(f"Value shape: {v.shape}")             # Should be [1, 1]
        logger.info(f"Policy logits: {p_logits.cpu().numpy()}")
        logger.info(f"Value: {v.item()}")

        # Test action selection (direct from network)
        action, probs = select_action_from_net(model, env, dev) # <<< Pass device
        logger.info(f"Selected action (from net): {action}")
        logger.info(f"Action probabilities: {probs}")

    except Exception as e:
        logger.exception(f"Error during example usage: {e}")