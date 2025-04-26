import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader # Import DataLoader
import torch.multiprocessing as mp
import torch.optim.lr_scheduler as lr_scheduler # Import scheduler
import random
from collections import deque
import time
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

# Setup logger
logger = get_logger("AlphaZeroTraining", "AlphaZeroTraining.log")

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
        state, policy, value = self.buffer[idx]
        # State should already be tensor on CPU, policy tensor, value float
        # Convert policy and value to tensors here for the DataLoader
        return state, policy , torch.tensor(value, dtype=torch.float32)

    def add(self, state, policy, value):
        # Add a new experience tuple (state_tensor_cpu, policy_numpy, value_float)
        policy_tensor = torch.from_numpy(policy).float()
        self.buffer.append((state.cpu(), policy_tensor, value))


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
        observation = env.reset()

        game_states = []
        game_policies = []

        temperature = params['temperature_init']
        move_count = 0

        while not env.done:
            if move_count >= params['temp_decay_steps'] and temperature > params['temperature_final']:
                temperature *= params['temperature_decay_factor']

            state_tensor_gpu = cxnn.preprocess_input(env).to(device)

            # Pass the log_debug_messages flag to MCTS
            action, policy = mcts.select_action(
                root_env=env,
                model=model,
                n_simulations=params['n_simulations'],
                c_puct=params['c_puct'],
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
            # Store state (tensor), policy (numpy), value (float)
            examples.append((game_states[i], game_policies[i], float(player_perspective_value)))

        return examples

    except Exception as e:
        # Log the full traceback using the main logger instance
        logger.error(f"Error in self-play worker {worker_id}: {e}", exc_info=True)
        return None

def train_network(model, optimizer, scheduler, buffer, params, device, global_step_counter):
    import random

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

def evaluate_model(current_model, previous_model, num_games, device, params):
    """Evaluate current model against previous version on the specified device"""
    logger.info(f"Starting evaluation: {num_games} games...")
    current_model.eval()
    previous_model.eval()

    current_wins = 0
    previous_wins = 0
    draws = 0

    for game_idx in tqdm(range(num_games), desc="Evaluation Games"):
        env = make("connectx", debug=False)
        env.reset()

        if game_idx % 2 == 0:
            model_p1, model_p2 = current_model, previous_model
            p1_is_current = True
        else:
            model_p1, model_p2 = previous_model, current_model
            p1_is_current = False

        while not env.done:
            current_player_idx = env.state[0]['observation']['mark'] - 1
            active_model = model_p1 if current_player_idx == 0 else model_p2

            with torch.no_grad():
                 # Use MCTS for evaluation (potentially stronger but slower)
                 # Set log_debug=False during evaluation to avoid excessive logs
                 action, _ = mcts.select_action(
                     root_env=env,
                     model=active_model,
                     n_simulations=params.get('eval_n_simulations', 50), # Fewer sims for eval
                     c_puct=params['c_puct'],
                     temperature=0, # Greedy selection during evaluation
                     device=device,
                     log_debug=False # Disable debug logs during evaluation
                 )
                 if action is None:
                      logger.error("MCTS returned None action during evaluation. Choosing random valid.")
                      valid_actions = [c for c in range(env.configuration.columns) if env.observation.board[c] == 0]
                      action = random.choice(valid_actions) if valid_actions else 0


            # Step the environment
            env.step([int(action), int(action)])

        # Determine winner
        reward_p1 = env.state[0]['reward']
        reward_p2 = env.state[1]['reward']

        if reward_p1 == 1:
            if p1_is_current: current_wins += 1
            else: previous_wins += 1
        elif reward_p2 == 1:
            if p1_is_current: previous_wins += 1
            else: current_wins += 1
        else:
            draws += 1

    win_rate = (current_wins + 0.5 * draws) / max(1, num_games) # Avoid division by zero
    logger.info(f"Evaluation Result - Current Wins: {current_wins}, Previous Wins: {previous_wins}, Draws: {draws}, Win Rate: {win_rate:.4f}")
    return win_rate, current_wins, previous_wins, draws

def save_loss_history(loss_history, filename="loss_history.csv"):
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

    model = cxnn.ConnectXNet().to(device)
    try: 
        model.load_state_dict(torch.load("models/last_model.pth", map_location=device))
        logger.info("Loaded last model.")
    except FileNotFoundError:
        logger.warning("Last model not found. Starting training from scratch.")
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAINING_PARAMS['learning_rate'],
        weight_decay=TRAINING_PARAMS['weight_decay']
    )
    replay_buffer = ReplayBuffer(max_size=TRAINING_PARAMS['buffer_size'])

    # --- Learning Rate Scheduler Setup ---
    # Estimate total training steps for scheduler configuration
    estimated_batches_per_epoch = TRAINING_PARAMS['buffer_size'] // TRAINING_PARAMS['batch_size']
    # Ensure estimated_batches_per_epoch is at least 1 if buffer is smaller than batch size initially
    estimated_batches_per_epoch = max(1, estimated_batches_per_epoch)
    total_steps = TRAINING_PARAMS['num_iterations'] * TRAINING_PARAMS['num_epochs'] * estimated_batches_per_epoch
    warmup_steps = TRAINING_PARAMS['warmup_steps']
    cosine_steps = max(1, total_steps - warmup_steps) # Ensure > 0
    initial_lr = TRAINING_PARAMS['learning_rate']
    min_lr = TRAINING_PARAMS['learning_rate'] * 0.05 # Minimum learning rate

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

    best_win_rate = 0.55 # Threshold to beat previous best

    # --- Training Loop ---
    for iteration in range(TRAINING_PARAMS['num_iterations']):
        iteration_start_time = time.time()
        iter_num = iteration + 1
        logger.info(f"===== Starting Iteration {iter_num}/{TRAINING_PARAMS['num_iterations']} =====")
        logger.info(f"Current Global Step: {global_step_counter[0]}, Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        if iteration % 2 == 0 and iteration > 0 :
            if TRAINING_PARAMS['num_self_play_games'] < TRAINING_PARAMS['num_self_play_games_limit']:
                TRAINING_PARAMS['num_self_play_games'] += 1 # Increment number of self-play games for next iteration


        # --- Self-Play Phase ---
        self_play_start_time = time.time()
        logger.info(f"Starting self-play with {TRAINING_PARAMS['num_workers']} workers...")
        model.eval() # Ensure model is in eval mode for self-play inference

        current_model_state_dict = copy.deepcopy(model.state_dict())
        for key in current_model_state_dict:
             current_model_state_dict[key] = current_model_state_dict[key].cpu()
        
        if iteration == 20:
            TRAINING_PARAMS['n_simulations'] = 110 # look deeper after early-training
        elif iteration == 40:
            TRAINING_PARAMS['n_simulations'] = 120
        elif iteration == 60:
            TRAINING_PARAMS['n_simulations'] = 130
        elif iteration == 80:
            TRAINING_PARAMS['n_simulations'] = 140
        elif iteration == 100:
            TRAINING_PARAMS['n_simulations'] = 150

        # Include worker_id in arguments passed to the pool
        worker_args = [(current_model_state_dict, TRAINING_PARAMS, str(device), i)
                       for i in range(TRAINING_PARAMS['num_self_play_games'])]

        all_game_examples = []
        try:
            with mp.Pool(processes=TRAINING_PARAMS['num_workers']) as pool: # Use default context or the one set globally
                results = list(tqdm(pool.imap_unordered(run_self_play_game, worker_args),
                                    total=TRAINING_PARAMS['num_self_play_games'],
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
                model, optimizer, scheduler, replay_buffer,
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
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        # # --- Evaluation Phase ---
        # if iter_num % TRAINING_PARAMS['eval_interval'] == 0 and iter_num > 0:
        #     eval_start_time = time.time()
        #     logger.info("Starting evaluation against previous best model...")

        #     previous_model = cxnn.ConnectXNet().to(device)
        #     best_model_path = "models/best/best_model.pth"

        #     if os.path.exists(best_model_path):
        #          try:
        #             previous_model_state_dict = torch.load(best_model_path, map_location=device)
        #             previous_model.load_state_dict(previous_model_state_dict)
        #             logger.info(f"Loaded previous best model from {best_model_path}")

        #             win_rate, _, _, _ = evaluate_model(
        #                 current_model=model,
        #                 previous_model=previous_model,
        #                 num_games=TRAINING_PARAMS['eval_games'],
        #                 device=device,
        #                 params=TRAINING_PARAMS
        #             )

        #             if win_rate > best_win_rate:
        #                 logger.info(f"New best model! Win rate: {win_rate:.4f} > {best_win_rate:.4f}")
        #                 best_win_rate = win_rate
        #                 best_path = "models/best/best_model.pth"
        #                 torch.save(model.state_dict(), best_path)
        #                 logger.info(f"Saved new best model to: {best_path}")
        #             else:
        #                 logger.info(f"Did not surpass best model. Win rate: {win_rate:.4f}, Best: {best_win_rate:.4f}")
        #                 # Optionally revert to the best model's weights if performance degrades
        #                 # model.load_state_dict(previous_model_state_dict)
        #                 # logger.info("Reverted to previous best model weights.")

        #          except Exception as e:
        #             logger.error(f"Failed evaluation loading previous model: {e}.", exc_info=True)
        #     else:
        #          logger.warning("No previous best model found. Saving current as best.")
        #          best_path = "models/best/best_model.pth"
        #          torch.save(model.state_dict(), best_path)
        #          logger.info(f"Saved initial best model to: {best_path}")


        #     eval_duration = time.time() - eval_start_time
        #     logger.info(f"Evaluation completed in {eval_duration:.2f}s.")

        iteration_duration = time.time() - iteration_start_time
        logger.info(f"Iteration {iter_num} finished in {iteration_duration:.2f}s.")
        # Estimate remaining time
        remaining_iters = TRAINING_PARAMS['num_iterations'] - iter_num
        if remaining_iters > 0:
             est_remaining_time = iteration_duration * remaining_iters
             logger.info(f"Estimated time remaining: {est_remaining_time:.2f}s ({est_remaining_time/3600:.2f} hours)")


    # --- Save Final Model & Loss History ---
    final_path = "models/final_model.pth"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Training completed. Final model saved at: {final_path}")

    # Save the collected loss history
    save_loss_history(loss_history, "results/loss_history.csv")

if __name__ == "__main__":
    # Ensure the script can be run directly, especially for multiprocessing.
    # mp.freeze_support() # Might be needed on Windows

    main()
