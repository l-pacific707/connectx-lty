import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader # Import DataLoader
import torch.multiprocessing as mp
import torch.optim.lr_scheduler as lr_scheduler # Import scheduler
from collections import deque
import time
import pickle
import os
import logging
from kaggle_environments import make
from tqdm import tqdm
import copy
import math # Import math for scheduler
import csv # Import csv for saving loss history
import yaml
# Import your existing modules
import MCTS_Connectx as mcts
import ConnectXNN as cxnn
from logger_setup import get_logger
from rng_init import rng_worker_init
from random import choice

EMPTY = 0

# Setup logger
logger = get_logger("train.py", "Play_and_Train.log")

# Training Parameters (Consider adjusting based on parallel execution)
with open("training_config.yaml", "r") as file:
    TRAINING_PARAMS = yaml.safe_load(file)



class ReplayBuffer(Dataset):
    """Stores self-play game data and acts as a PyTorch Dataset."""
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        # Retrieve an item by index. Ensure data types are consistent.
        state_np, policy, value = self.buffer[idx]
        # State should already be tensor on CPU, policy tensor, value float
        # Convert policy and value to tensors here for the DataLoader
        return state_np, policy , torch.tensor(value, dtype=torch.float32)

    def add(self, state_np, policy, value):
        # Add a new experience tuple (state_tensor_cpu, policy_numpy, value_float)
        policy_tensor = torch.from_numpy(policy).float()
        self.buffer.append((state_np, policy_tensor, value))

def flip_board_state_numpy(state_np):
    """
    Flips the board state numpy array horizontally.
    Assumes state_np shape is (C, H, W), e.g., (3, 6, 7).
    """
    # Flip along the last axis (axis=2, which corresponds to Width)
    if state_np.ndim == 3:
        # Flip the state horizontally
        return np.flip(state_np, axis=2).copy()
    elif state_np.ndim == 4:
        # Flip the state horizontally and change the channel order
        # Assuming state_np shape is (B, C, H, W)
        # Flip along the last axis (axis=3, which corresponds to Width)
        return np.flip(state_np, axis=3).copy()
    logger.warning(f"Unexpected state shape: {state_np.shape}. Expected 3D or 4D.")
    return state_np # Return unchanged if not 3D or 4D


def flip_policy_vector(policy_vector):
    """
    Flips the policy vector horizontally.
    Assumes policy_vector is a 1D numpy array of length num_actions.
    """
    # Flips the order, e.g., for 7 actions [0,1,2,3,4,5,6] -> [6,5,4,3,2,1,0]
    return np.flip(policy_vector, axis=0)

# Function to run a single self-play game (designed for multiprocessing)
def run_self_play_game(args):
    """
    Executes one game of self-play. Designed to be run in a separate process.

    Args:
        args (tuple): Contains model_state_dict, params, device_str, worker_id.

    Returns:
        list: A list of tuples, where each tuple contains (state_tensor_cpu, policy_numpy, value_float)
              representing one step in the game. Returns None on error.
    """
    # Unpack arguments including worker_id
    model_state_dict, params, device_str, worker_id = args
    device = torch.device(device_str)

    # Determine if this worker should produce debug logs
    log_debug_messages = (worker_id == 0) # Only worker 0 logs debug messages
    from rng_init import np_rng, torch_rng

    # Optional: Reconfigure logger level for non-debug workers if needed
    # if not log_debug_messages:
    #     worker_logger = get_logger("AlphaZeroTraining") # Get logger instance in worker
    #     worker_logger.setLevel(logging.INFO) # Suppress debug messages for this worker

    try:
        # Create a new model instance in the subprocess
        model = cxnn.ConnectXNet()
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()

        examples = []
        env = make("connectx", debug=False) # Keep debug=False for performance

        game_states = []
        game_policies = []

        temperature = params['temperature_init']
        move_count = 0

        while not env.done:
            if move_count > params['temp_decay_steps'] and temperature > params['temperature_final']:
                temperature *= params['temperature_decay_factor']
            if move_count > params['noise_threshold']:
                params['mcts_alpha_std'] = 0.0 # Disable noise after threshold
                


            state_tensor_gpu = cxnn.preprocess_input(env).to(device)

            # Pass the log_debug_messages flag to MCTS
            action, policy = mcts.select_action(
                root_env=env,
                model=model,
                n_simulations=params['n_simulations'],
                c_puct=params['c_puct'],
                c_fpu=params['c_fpu'],
                mcts_alpha=params['mcts_alpha_std'],
                mcts_epsilon=params['mcts_epsilon'],
                np_rng= np_rng,
                temperature=temperature,
                device=device,
                log_debug=log_debug_messages # Pass the flag here
            )

            # Store state on CPU and policy as numpy array
            game_states.append(state_tensor_gpu.cpu())
            game_policies.append(policy) # policy is already numpy array

            if action is None:
                 # Use logger.error which should always be visible
                 logger.error(f"Worker {worker_id}: MCTS returned None action during self-play. Ending game.")
                 break # End game if MCTS fails

            env.step([int(action), int(action)])
            move_count += 1

        # Determine game result
        reward_p1 = env.state[0]['reward']
        reward_p2 = env.state[1]['reward']
        value = 1.0 if reward_p1 == 1 else (-1.0 if reward_p2 == 1 else 0.0)

        # Assign values and add examples
        for i in range(len(game_states)):
            player_perspective_value = value if i % 2 == 0 else -value
            state_np = game_states[i].numpy()
            # Store state (numpy array), policy (numpy), value (float)
            examples.append((state_np, game_policies[i], float(player_perspective_value)))
            # Data augmentation: flip the board and policy
            examples.append((flip_board_state_numpy(state_np), flip_policy_vector(game_policies[i]), float(player_perspective_value)))

        return examples

    except Exception as e:
        # Log the full traceback using the main logger instance
        logger.error(f"Error in self-play worker {worker_id}: {e}", exc_info=True)
        return None

def train_network(model, optimizer, scheduler, buffer, params, device, global_step_counter):
    """_summary_

    Args:
        model (_type_): _description_
        optimizer (_type_): _description_
        scheduler (_type_): _description_
        buffer (_type_): _description_
        params (_type_): _description_
        device (_type_): _description_
        global_step_counter (_type_): _description_

    Returns:
        _type_: _description_
    """    

    logger.info(f"Starting training on device: {device} with {len(buffer)} examples.")
    model.train()
    
    dataloader = DataLoader(
        buffer,
        batch_size=params['batch_size'],
        shuffle=True, num_workers = 0,
        pin_memory=torch.cuda.is_available() # Pin memory for faster data transfer to GPU
    )
    

    total_loss_accum = 0.0
    policy_loss_accum = 0.0
    value_loss_accum = 0.0
    batches_processed = 0



    for epoch in range(params['num_epochs']):
        epoch_desc = f"Epoch {epoch+1}/{params['num_epochs']}"
        for states, policies, values in tqdm(dataloader, desc=epoch_desc, leave=False):
            states = states.to(device, non_blocking=True)
            policies = policies.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True).unsqueeze(1)
            
            if states.dim() == 5 and states.shape[1] == 1:
                states = states.squeeze(1)
            elif states.dim() != 4:
                logger.error(f"Unexpected state tensor shape: {states.shape}")
                continue

            optimizer.zero_grad()

            policy_logits, value_pred = model(states)

            value_loss = F.mse_loss(value_pred, values)
            policy_loss = -torch.sum(policies * F.log_softmax(policy_logits, dim=1), dim=1).mean()
            loss = value_loss + policy_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step_counter[0] += 1

            total_loss_accum += loss.item()
            policy_loss_accum += policy_loss.item()
            value_loss_accum += value_loss.item()
            batches_processed += 1

            if global_step_counter[0] % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.debug(f"Step: {global_step_counter[0]}, LR: {current_lr:.6f}")



    if batches_processed == 0:
        logger.warning("No batches processed.")
        return 0.0, 0.0, 0.0
    else:
        avg_total_loss = total_loss_accum / batches_processed
        avg_policy_loss = policy_loss_accum / batches_processed
        avg_value_loss = value_loss_accum / batches_processed
        logger.info(f"Training Avg Losses - Total: {avg_total_loss:.4f}, Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f}")
        return avg_total_loss, avg_policy_loss, avg_value_loss



## Auxilary agent to evaluate the model
def play(board, column, mark, config):
    columns = config.columns
    rows = config.rows
    row = max([r for r in range(rows) if board[column + (r * columns)] == EMPTY])
    board[column + (row * columns)] = mark


def is_win(board, column, mark, config, has_played=True):
    columns = config.columns
    rows = config.rows
    inarow = config.inarow - 1
    row = (
        min([r for r in range(rows) if board[column + (r * columns)] == mark])
        if has_played
        else max([r for r in range(rows) if board[column + (r * columns)] == EMPTY])
    )

    def count(offset_row, offset_column):
        for i in range(1, inarow + 1):
            r = row + offset_row * i
            c = column + offset_column * i
            if (
                r < 0
                or r >= rows
                or c < 0
                or c >= columns
                or board[c + (r * columns)] != mark
            ):
                return i - 1
        return inarow

    return (
        count(1, 0) >= inarow  # vertical.
        or (count(0, 1) + count(0, -1)) >= inarow  # horizontal.
        or (count(-1, -1) + count(1, 1)) >= inarow  # top left diagonal.
        or (count(-1, 1) + count(1, -1)) >= inarow  # top right diagonal.
    )


def negamax_agent(obs, config):
    columns = config.columns
    rows = config.rows
    size = rows * columns

    # Due to compute/time constraints the tree depth must be limited.
    max_depth = 4

    def negamax(board, mark, depth):
        moves = sum(1 if cell != EMPTY else 0 for cell in board)

        # Tie Game
        if moves == size:
            return (0, None)

        # Can win next.
        for column in range(columns):
            if board[column] == EMPTY and is_win(board, column, mark, config, False):
                return ((size + 1 - moves) / 2, column)

        # Recursively check all columns.
        best_score = -size
        best_column = None
        for column in range(columns):
            if board[column] == EMPTY:
                # Max depth reached. Score based on cell proximity for a clustering effect.
                if depth <= 0:
                    row = max(
                        [
                            r
                            for r in range(rows)
                            if board[column + (r * columns)] == EMPTY
                        ]
                    )
                    score = (size + 1 - moves) / 2
                    if column > 0 and board[row * columns + column - 1] == mark:
                        score += 1
                    if (
                        column < columns - 1
                        and board[row * columns + column + 1] == mark
                    ):
                        score += 1
                    if row > 0 and board[(row - 1) * columns + column] == mark:
                        score += 1
                    if row < rows - 2 and board[(row + 1) * columns + column] == mark:
                        score += 1
                else:
                    next_board = board[:]
                    play(next_board, column, mark, config)
                    (score, _) = negamax(next_board,
                                         1 if mark == 2 else 2, depth - 1)
                    score = score * -1
                if score > best_score or (score == best_score and choice([True, False])):
                    best_score = score
                    best_column = column

        return (best_score, best_column)

    _, column = negamax(obs.board[:], obs.mark, max_depth)
    if column == None:
        column = choice([c for c in range(columns) if obs.board[c] == EMPTY])
    return column

# Worker function for parallel evaluation
def run_single_evaluation_game_worker(args):
    current_model_state_dict, previous_model_state_dict, device_str, params, worker_id = args
    device = torch.device(device_str)

    # RNG is initialized by the pool's initializer (rng_worker_init)
    from rng_init import np_rng

    log_debug_eval = False # Typically no debug logging for evaluation games

    current_model_worker = cxnn.ConnectXNet().to(device)
    current_model_worker.load_state_dict(current_model_state_dict)
    current_model_worker.eval()

    previous_model_worker = cxnn.ConnectXNet().to(device)
    previous_model_worker.load_state_dict(previous_model_state_dict)
    previous_model_worker.eval()
    
    env = make("connectx", debug=False)
    env.reset()

    # Alternate who starts
    if worker_id % 2 == 0:
        model_p1, model_p2 = current_model_worker, previous_model_worker
        p1_is_current = True
    else:
        model_p1, model_p2 = previous_model_worker, current_model_worker
        p1_is_current = False

    while not env.done:
        if env.state[0]["status"] == "ACTIVE":
                current_player_mark = 1
        elif env.state[1]["status"] == "ACTIVE":
                current_player_mark = 2
        active_model = model_p1 if current_player_mark == 1 else model_p2
        
        # Use worker-specific np_rng for MCTS
        action, _ = mcts.select_action(
            root_env=env,
            model=active_model,
            n_simulations=params['n_simulations_eval'],
            c_puct=params['c_puct'],
            c_fpu=0.0, # Typically FPU is not heavily used or is 0 in eval for greedy play
            mcts_alpha=0.0, # No noise in evaluation
            mcts_epsilon=0.0, # No noise
            np_rng=np_rng, # Pass the worker-specific RNG
            temperature=0.0, # Greedy action selection
            device=device,
            log_debug=log_debug_eval
        )

        if action is None:
            logger.error(f"Eval Worker {worker_id}: MCTS returned None. Game cannot proceed.")
            # This might be a draw or an issue. Let's treat as draw for robustness.
            return 0, p1_is_current # (0 for draw, bool indicating if P1 was current)
        
        
        step_actions = [None, None]
        step_actions[current_player_mark -1] = int(action)
        env.step(step_actions)

    reward_p1 = env.state[0]['reward']
    if reward_p1 == 1: # P1 won
        return 1, p1_is_current # (1 for P1 win, bool)
    elif reward_p1 == -1: # P2 won (P1 lost)
        return -1, p1_is_current # (-1 for P2 win, bool)
    else: # Draw
        return 0, p1_is_current # (0 for draw, bool)

def run_single_evaluation_game_worker_against_negamax(args):
    current_model_state_dict, device_str, params, worker_id = args
    device = torch.device(device_str)

    # RNG is initialized by the pool's initializer (rng_worker_init)
    from rng_init import np_rng

    log_debug_eval = False # Typically no debug logging for evaluation games

    current_model_worker = cxnn.ConnectXNet().to(device)
    current_model_worker.load_state_dict(current_model_state_dict)
    current_model_worker.eval()

    previous_model_worker = None
    
    env = make("connectx", debug=False)
    env.reset()

    # Alternate who starts
    if worker_id % 2 == 0:
        model_p1, model_p2 = current_model_worker, previous_model_worker
        p1_is_current = True
    else:
        model_p1, model_p2 = previous_model_worker, current_model_worker
        p1_is_current = False

    while not env.done:
        if env.state[0]["status"] == "ACTIVE":
                current_player_mark = 1
        elif env.state[1]["status"] == "ACTIVE":
                current_player_mark = 2
        active_model = model_p1 if current_player_mark == 1 else model_p2
        
        # Use worker-specific np_rng for MCTS
        if active_model is not None:
            action, _ = mcts.select_action(
                root_env=env,
                model=active_model,
                n_simulations=params['n_simulations_eval'],
                c_puct=params['c_puct'],
                c_fpu=0.0, # Typically FPU is not heavily used or is 0 in eval for greedy play
                mcts_alpha=0.0, # No noise in evaluation
                mcts_epsilon=0.0, # No noise
                np_rng=np_rng, # Pass the worker-specific RNG
                temperature=0.0, # Greedy action selection
                device=device,
                log_debug=log_debug_eval
            )
        elif active_model is None:
            # Use negamax agent
            action = negamax_agent(env.state[0]['observation'], env.configuration)

        if action is None:
            logger.error(f"Eval Worker {worker_id}: MCTS returned None. Game cannot proceed.")
            # This might be a draw or an issue. Let's treat as draw for robustness.
            return 0, p1_is_current # (0 for draw, bool indicating if P1 was current)
        
        
        step_actions = [None, None]
        step_actions[current_player_mark -1] = int(action)
        env.step(step_actions)

    reward_p1 = env.state[0]['reward']
    if reward_p1 == 1: # P1 won
        return 1, p1_is_current # (1 for P1 win, bool)
    elif reward_p1 == -1: # P2 won (P1 lost)
        return -1, p1_is_current # (-1 for P2 win, bool)
    else: # Draw
        return 0, p1_is_current # (0 for draw, bool)



def evaluate_model(current_model_sd, previous_model_sd, num_games, device_str, params, num_workers):
    logger.info(f"Starting parallel self-evaluation: {num_games} games with {num_workers} workers...")
    
    current_wins = 0
    previous_wins = 0
    draws = 0

    # Ensure state dicts are on CPU before sending to workers
    current_model_cpu_sd = {k: v.cpu() for k, v in current_model_sd.items()}
    previous_model_cpu_sd = {k: v.cpu() for k, v in previous_model_sd.items()}

    worker_args_list = [
        (current_model_cpu_sd, previous_model_cpu_sd, device_str, params, game_idx)
        for game_idx in range(num_games)
    ]
    
    # Use a base seed for evaluation workers, ensuring variety if num_games > num_workers
    eval_base_seed = params['base_seed'] + 1000 # Offset from self-play seed

    with mp.Pool(processes=num_workers, initializer=rng_worker_init, initargs=(eval_base_seed,)) as pool:
        results = list(tqdm(pool.imap_unordered(run_single_evaluation_game_worker, worker_args_list),
                            total=num_games, desc="Evaluation Games"))

    for game_result, p1_was_current_model in results:
        if game_result == 1: # P1 won
            if p1_was_current_model: current_wins +=1
            else: previous_wins += 1
        elif game_result == -1: # P2 won
            if p1_was_current_model: previous_wins += 1
            else: current_wins += 1
        else: # Draw
            draws +=1
            
    total_games_played = current_wins + previous_wins + draws
    if total_games_played == 0: # Should not happen if num_games > 0
        win_rate = 0.0
        logger.warning("No games played. Win rate is 0.0.")
    else:
        # Win rate: (wins by current + 0.5 * draws) / total valid games for current
        # This definition needs care. Let's use the standard:
        # (current_model_wins + 0.5 * draws_where_current_played) / games_current_played
        # Simpler: (total wins for current + 0.5 * total draws) / total games
        win_rate = (current_wins + 0.5 * draws) / max(1, num_games)

    logger.info(f"Self Evaluation Result - Current Wins: {current_wins}, Previous (best) Wins: {previous_wins}, Draws: {draws}, Win Rate for Current Model: {win_rate:.4f}")
    
    
    return win_rate, current_wins, previous_wins, draws

def evaluate_model_against_negamax(current_model_sd, num_games, device_str, params, num_workers):
    logger.info(f"Starting parallel evaluation against negamax(4): {num_games} games with {num_workers} workers...")
    
    current_wins = 0
    previous_wins = 0
    draws = 0

    # Ensure state dicts are on CPU before sending to workers
    current_model_cpu_sd = {k: v.cpu() for k, v in current_model_sd.items()}

    worker_args_list = [
        (current_model_cpu_sd, device_str, params, game_idx)
        for game_idx in range(num_games)
    ]
    
    # Use a base seed for evaluation workers, ensuring variety if num_games > num_workers
    eval_base_seed = params['base_seed'] + 1000 # Offset from self-play seed

    with mp.Pool(processes=num_workers, initializer=rng_worker_init, initargs=(eval_base_seed,)) as pool:
        results = list(tqdm(pool.imap_unordered(run_single_evaluation_game_worker_against_negamax, worker_args_list),
                            total=num_games, desc="Evaluation Games"))

    for game_result, p1_was_current_model in results:
        if game_result == 1: # P1 won
            if p1_was_current_model: current_wins +=1
            else: previous_wins += 1
        elif game_result == -1: # P2 won
            if p1_was_current_model: previous_wins += 1
            else: current_wins += 1
        else: # Draw
            draws +=1
            
    total_games_played = current_wins + previous_wins + draws
    if total_games_played == 0: # Should not happen if num_games > 0
        win_rate = 0.0
        logger.warning("No games played. Win rate is 0.0.")
    else:
        # Win rate: (wins by current + 0.5 * draws) / total valid games for current
        # This definition needs care. Let's use the standard:
        # (current_model_wins + 0.5 * draws_where_current_played) / games_current_played
        # Simpler: (total wins for current + 0.5 * total draws) / total games
        win_rate = (current_wins + 0.5 * draws) / max(1, num_games)

    logger.info(f"Evaluation against negamax Agent Result - Current Wins: {current_wins}, Previous (best) Wins: {previous_wins}, Draws: {draws}, Win Rate for Current Model: {win_rate:.4f}")
    
    return win_rate, current_wins, previous_wins, draws


def save_replay_buffer(buffer: ReplayBuffer, path: str):
    data_to_save = []

    for state_np, policy_tensor, value_float in buffer.buffer:
        # policy tensor를 numpy로 변환
        policy_np = policy_tensor.cpu().numpy()
        data_to_save.append((state_np, policy_np, value_float))

    with open(path, 'wb') as f:
        pickle.dump(data_to_save, f)

def load_replay_buffer(path: str, buffer: ReplayBuffer):
    with open(path, 'rb') as f:
        data_loaded = pickle.load(f)

    for state_np, policy_np, value_float in data_loaded:
        buffer.add(state_np, policy_np, value_float)


def save_loss_history(loss_history, filename="results/loss_history.csv"):
    """Saves the collected loss history to a CSV file."""
    try:
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['iteration', 'avg_total_loss', 'avg_policy_loss', 'avg_value_loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i, losses in enumerate(loss_history):
                writer.writerow({
                    'iteration': i + 1,
                    'avg_total_loss': losses[0],
                    'avg_policy_loss': losses[1],
                    'avg_value_loss': losses[2]
                })
        logger.info(f"Loss history saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save loss history: {e}", exc_info=True)


def main():
    # --- Setup ---
    try:
        # Try setting the start method to 'spawn' for CUDA compatibility.
        # This is often necessary on macOS and Windows.
        if mp.get_start_method(allow_none=True) is None:
             mp.set_start_method('spawn')
             logger.info("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        logger.warning("Multiprocessing start method already set or failed to set.")
        pass
    except AttributeError:
         logger.warning("mp.get_start_method not available. Skipping start method setting (might be ok on Linux).")


    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    model_train = cxnn.ConnectXNet().to(device)
    model_play = cxnn.ConnectXNet().to(device)
    
    # Load best model state (for model_play and as initial best)
    best_model_path = "models/best/best_model.pth"
    try: 
        model_train.load_state_dict(torch.load("models/last_model.pth", map_location=device))
        logger.info("Loaded last model.")
    except FileNotFoundError:
        logger.warning("last model not found. Starting training from scratch.")
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        
    try:
        model_play.load_state_dict(torch.load("models/best/best_model.pth", map_location=device))
        logger.info("Loaded best model.")
    except FileNotFoundError:
        logger.warning("best model not found. Try to recover it from the last model.")
        try:
            model_play.load_state_dict(torch.load("models/last_model.pth", map_location=device))
            logger.info("Loaded last model as best model.")
        except FileNotFoundError:
            logger.warning("last model not found. Starting training from scratch.")
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True) 
        
        
    optimizer = optim.AdamW(
        model_train.parameters(),
        lr=TRAINING_PARAMS['learning_rate'],
        weight_decay=TRAINING_PARAMS['weight_decay']
    )
    replay_buffer = ReplayBuffer(max_size=TRAINING_PARAMS['buffer_size'])
    
    # load previous buffers
    try:
        load_replay_buffer("models/replay_buffer.pkl", replay_buffer)
        logger.info("Loaded replay buffer.")
    except Exception as e:
        logger.info("There's no existing replay buffer: {e}", exc_info=True)

    # --- Learning Rate Scheduler Setup ---
    # Estimate total training steps for scheduler configuration
    estimated_batches_per_epoch = TRAINING_PARAMS['buffer_size'] // TRAINING_PARAMS['batch_size']
    # Ensure estimated_batches_per_epoch is at least 1 if buffer is smaller than batch size initially
    estimated_batches_per_epoch = max(1, estimated_batches_per_epoch)
    total_steps = TRAINING_PARAMS['num_iterations'] * TRAINING_PARAMS['num_epochs'] * estimated_batches_per_epoch
    warmup_steps = int(TRAINING_PARAMS['warmup_steps_ratio'] * total_steps)
    cosine_steps = max(1, total_steps - warmup_steps) # Ensure > 0
    initial_lr = TRAINING_PARAMS['learning_rate']
    min_lr = TRAINING_PARAMS['learning_rate'] * 0.05 # Minimum rate

    logger.info(f"Scheduler: Estimated total steps: {total_steps}, Warmup steps: {warmup_steps}")

    def lr_lambda(current_step):
        """ Lambda function for LR scheduling: linear warmup then cosine decay. """
        if current_step < warmup_steps:
            # Linear warmup phase
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay phase
            progress = float(current_step - warmup_steps) / float(max(1, cosine_steps))
            # Ensure progress doesn't exceed 1.0 for cosine calculation
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Scale cosine decay from 1.0 -> min_lr/initial_lr
            scale = (1.0 - min_lr / initial_lr) * cosine_decay + (min_lr / initial_lr)
            return scale

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # --- Global Step Counter & Loss History ---
    global_step_counter = [0] # Use a list to make it mutable across function calls
    loss_history = [] # List to store (total_loss, policy_loss, value_loss) per iteration

    os.makedirs("models", exist_ok=True)
    os.makedirs("models/best", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    win_rate_threshold = 0.5 # Threshold to beat previous best
    base_seed = TRAINING_PARAMS["base_seed"]
    all_start_time = time.time()

    # --- Training Loop ---
    for iteration in range(TRAINING_PARAMS['num_iterations']):
        iteration_start_time = time.time()
        iter_num = iteration + 1
        logger.info(f"===== Starting Iteration {iter_num}/{TRAINING_PARAMS['num_iterations']} =====")
        logger.info(f"Current Global Step: {global_step_counter[0]}, Current LR: {optimizer.param_groups[0]['lr']*(10**5):.6f}e-5")



        # --- Self-Play Phase ---
        self_play_start_time = time.time()
        logger.info(f"Starting self-play with {TRAINING_PARAMS['num_workers']} workers...")
        model_play.eval() # Ensure model is in eval mode for self-play inference

        current_model_state_dict = copy.deepcopy(model_play.state_dict())
        for key in current_model_state_dict:
             current_model_state_dict[key] = current_model_state_dict[key].cpu()
        
        # if iteration % 20 == 0 and iteration > 0:
        #     # Increase the number of simulations for deeper exploration
        #     TRAINING_PARAMS['n_simulations'] += 10 # look deeper after early-training


        # Include worker_id in arguments passed to the pool
        worker_args = [(current_model_state_dict, TRAINING_PARAMS, str(device), i)
                       for i in range(TRAINING_PARAMS['num_self_play_games_per_worker']* TRAINING_PARAMS['num_workers'])]

        all_game_examples = []
        try:
            with mp.Pool(
                processes=TRAINING_PARAMS['num_workers'],
                initializer=rng_worker_init,
                initargs=((base_seed + iter_num),)
                ) as pool: # Use default context or the one set globally
                results = list(tqdm(pool.imap_unordered(run_self_play_game, worker_args),
                                    total=TRAINING_PARAMS['num_self_play_games_per_worker'] * TRAINING_PARAMS['num_workers'],
                                    desc=f"Iter {iter_num} Self-Play"))
                for game_examples in results:
                    if game_examples is not None:
                        all_game_examples.extend(game_examples)
                    else:
                        # Logged inside run_self_play_game now
                        logger.warning("A self-play worker returned an error (None).")
        except Exception as e:
             logger.error(f"Error during parallel self-play: {e}", exc_info=True)

        self_play_duration = time.time() - self_play_start_time
        logger.info(f"Self-play completed in {self_play_duration:.2f}s. Generated {len(all_game_examples)} examples.")

        if all_game_examples:
            add_start_time = time.time()
            for state, policy, value in all_game_examples:
                replay_buffer.add(state, policy, value)
            add_duration = time.time() - add_start_time
            logger.info(f"Added examples to buffer (Size: {len(replay_buffer)}/{TRAINING_PARAMS['buffer_size']}) in {add_duration:.2f}s.")
        else:
             logger.warning("No examples generated from self-play.")
             # Decide if training should proceed without new data
             # if len(replay_buffer) < TRAINING_PARAMS['batch_size']: continue


        # --- Training Phase ---
        avg_iter_losses = (0.0, 0.0, 0.0) # Default losses if training skipped
        if len(replay_buffer) >= TRAINING_PARAMS['batch_size']:
            train_start_time = time.time()
            avg_iter_losses = train_network(
                model_train, optimizer, scheduler, replay_buffer,
                TRAINING_PARAMS, device, global_step_counter
            )
            train_duration = time.time() - train_start_time
            logger.info(f"Training completed in {train_duration:.2f}s.")
        else:
            logger.info("Skipping training: not enough data in buffer.")

        # Store average losses for this iteration
        loss_history.append(avg_iter_losses)

        # --- Save Checkpoint ---
        if iter_num % TRAINING_PARAMS['checkpoint_interval'] == 0 or iter_num == TRAINING_PARAMS['num_iterations']:
            checkpoint_path = f"models/checkpoints/model_iter_{iter_num}.pth"
            torch.save(model_train.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            final_path = "models/last_model.pth"
            torch.save(model_train.state_dict(), final_path)
            logger.info(f"Saved last model: {final_path}")
            save_replay_buffer(replay_buffer, "models/replay_buffer.pkl")
            logger.info(f"Saved replay buffer. length: {len(replay_buffer)}")

        # --- Evaluation Phase ---
        if iter_num % TRAINING_PARAMS['eval_interval'] == 0 and iter_num > 0:
            eval_start_time = time.time()
            logger.info("Starting evaluation against previous best model...")
            
            # model_train is the current challenger, model_play is the current champion
            win_rate1, _, _, _ = evaluate_model(
                current_model_sd=model_train.state_dict(), # Challenger's state dict
                previous_model_sd=model_play.state_dict(), # Champion's state dict
                num_games=TRAINING_PARAMS['eval_games_per_worker'] * TRAINING_PARAMS['num_workers'],
                device_str=str(device),
                params=TRAINING_PARAMS,
                num_workers=TRAINING_PARAMS['num_workers'] # Use same number of workers for eval
            )
            
            win_rate2, _, _, _ = evaluate_model_against_negamax(
                current_model_sd=model_train.state_dict(), # Challenger's state dict
                num_games=TRAINING_PARAMS['eval_games_per_worker'] * TRAINING_PARAMS['num_workers'],
                device_str=str(device),
                params=TRAINING_PARAMS,
                num_workers=TRAINING_PARAMS['num_workers'] # Use same number of workers for eval
            )
            

            if win_rate1 >= 0.5 and win_rate2 >= win_rate_threshold:
                logger.info(f"Current model surpassed best model with  win rate: [{win_rate1:.4f}(self-play with best model) vs {win_rate2:.4f}(play against negamax(4) agent)]")
                win_rate_threshold = win_rate2 # Update threshold for next eval
                torch.save(model_train.state_dict(), best_model_path) # Save challenger as new best
                model_play.load_state_dict(model_train.state_dict()) # Update champion
                logger.info(f"Saved new best model to: {best_model_path}")
                #truncate the replay buffer to 15% of its size
                current_length = len(replay_buffer.buffer)
                num_to_keep = int(current_length * 0.15)
                num_to_remove = current_length - num_to_keep
                if num_to_remove > 0:
                    for _ in range(num_to_remove):
                        if len(replay_buffer.buffer) > 0: # Check before popping
                            replay_buffer.buffer.popleft()
                        else:
                            # This should ideally not be reached if num_to_remove is calculated correctly
                            # from current_length and the buffer isn't modified elsewhere concurrently.
                            break 
                    logger.info(f"Replay buffer truncated by popping {num_to_remove} elements. New size: {len(replay_buffer.buffer)}")
            else:
                logger.info(f"Current model did not surpass best model. Win rate: {win_rate1:.4f}(self-play), {win_rate2:.4f}(play against negamax(4)). Threshold : {win_rate_threshold:.4f}")
                # model_train continues to train, model_play remains the old best.
            eval_duration = time.time() - eval_start_time
            logger.info(f"Evaluation completed in {eval_duration:.2f}s.")


    # --- Save Final Model & Loss History ---
    final_path = "models/last_model.pth"
    torch.save(model_train.state_dict(), final_path)
    logger.info(f"Training completed. Final model saved at: {final_path}")

    # Save the collected loss history
    save_loss_history(loss_history, "results/loss_history.csv")
    
    
    total_duration = time.time() - all_start_time
    logger.info(f"Total training time: {total_duration:.2f}s ({total_duration/3600:.2f} hours)")


if __name__ == "__main__":
    # Ensure the script can be run directly, especially for multiprocessing.
    # Load the configuration file
    main()
