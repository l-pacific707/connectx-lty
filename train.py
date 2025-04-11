import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp # Import multiprocessing
import random
from collections import deque
import time
import os
import logging
from kaggle_environments import make
from tqdm import tqdm # Import tqdm
import copy # Import copy for deep copying model state

# Import your existing modules
import MCTS_Connectx as mcts
import ConnectXNN as cxnn
from logger_setup import get_logger

# Setup logger
logger = get_logger("AlphaZeroTraining", "AlphaZeroTraining.log")

# Training Parameters (Consider adjusting based on parallel execution)
TRAINING_PARAMS = {
    # MCTS parameters
    'n_simulations': 10,       # Number of MCTS simulations per move (Increased for potentially stronger play)
    'c_puct': 1.5,              # Exploration constant for MCTS 

    # Self-play parameters
    'num_self_play_games': 200, # Number of self-play games per iteration (Can increase with parallelism)
    'temperature_init': 1.0,    # Initial temperature for action selection
    'temperature_decay_factor' : 0.95, # Decay factor for temperature
    'temperature_final': 0.1,   # Final temperature after temp_decay_steps
    'temp_decay_steps': 10,     # Number of moves before temperature decay
    'num_workers': os.cpu_count() // 2, # Number of parallel workers for self-play (Adjust as needed)

    # Training parameters
    'batch_size': 256,          # Training batch size (Can increase with GPU)
    'buffer_size': 20000,       # Maximum size of replay buffer (Increased)
    'num_epochs': 5,            # Epochs per training iteration (Adjust as needed)
    'learning_rate': 0.0005,    # Learning rate 
    'weight_decay': 1e-4,       # L2 regularization parameter

    # Checkpoint parameters
    'checkpoint_interval': 10,  # Save model every n iterations
    'eval_interval': 5,         # Evaluate against previous version every n iterations
    'eval_games': 20,           # Number of evaluation games (Increased for reliability)

    # Training iterations
    'num_iterations': 100       # Total number of iterations (Increased)
}

class ReplayBuffer(Dataset):
    """Stores self-play game data."""
    def __init__(self, max_size=10000):
        # Use a deque for efficient addition and removal of old elements
        self.buffer = deque(maxlen=max_size)

    def __len__(self):
        # Return the current number of items in the buffer
        return len(self.buffer)

    def __getitem__(self, idx):
        # Retrieve an item by index (needed for Dataset compatibility)
        state, policy, value = self.buffer[idx]
        return state, policy, value

    def add(self, state, policy, value):
        # Add a new experience tuple (state, policy, value) to the buffer
        # Ensure tensors added are on CPU to avoid multiprocessing issues with CUDA tensors
        self.buffer.append((state.cpu(), policy, value))

    def sample(self, batch_size):
        """Samples a batch of experiences from the buffer."""
        # Ensure batch size is not larger than the buffer size
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        # Randomly sample indices
        indices = random.sample(range(len(self.buffer)), batch_size)
        # Retrieve samples using indices
        samples = [self.buffer[i] for i in indices]

        # Unzip the samples into separate lists
        states, policies, values = zip(*samples)

        # Stack the states into a single tensor
        # States are expected to be (1, C, H, W), stacking adds batch dim (B, 1,  C, H, W)
        states_tensor = torch.stack(states, dim=0).squeeze(1) # shape: (B, C, H, W)

        # Convert policies and values to tensors
        policies_tensor = torch.tensor(np.array(policies), dtype=torch.float32) # shape: (B, num_actions)
        values_tensor = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1) # shape: (B, 1)

        return states_tensor, policies_tensor, values_tensor

# Function to run a single self-play game (designed for multiprocessing)
def run_self_play_game(args):
    """
    Executes one game of self-play. Designed to be run in a separate process.

    Args:
        args (tuple): Contains model_state_dict, params, device.

    Returns:
        list: A list of tuples, where each tuple contains (state, policy, value)
              representing one step in the game. Returns None on error.
    """
    model_state_dict, params, device_str = args
    device = torch.device(device_str) # Recreate device object in the subprocess

    try:
        # Create a new model instance in the subprocess
        model = cxnn.ConnectXNet()
        model.load_state_dict(model_state_dict)
        model.to(device) # Move model to the specified device
        model.eval() # Set model to evaluation mode

        examples = []
        env = make("connectx", debug=False) # Turn off debug for performance in workers
        observation = env.reset()

        game_states = []
        game_policies = []

        temperature = params['temperature_init']
        move_count = 0

        while not env.done:
            # Decay temperature
            if move_count >= params['temp_decay_steps'] and temperature > params['temperature_final']:
                temperature *= params['temperature_decay_factor']

            # Get state representation and move to device
            state_tensor = cxnn.preprocess_input(env).to(device) # Move input to device

            # Get action and policy from MCTS
            # Ensure MCTS uses the model on the correct device
            action, policy = mcts.select_action(
                root_env=env,
                model=model, # Pass the model instance on the correct device
                n_simulations=params['n_simulations'],
                c_puct=params['c_puct'],
                temperature=temperature,
                device=device # Pass device to MCTS if it needs it for internal model calls
            )

            # Save state (on CPU) and policy (numpy array)
            game_states.append(state_tensor.cpu()) # Store state on CPU
            game_policies.append(policy) # Policy is usually a numpy array

            # Execute action (both players use the same policy)
            # Ensure action is an integer
            env.step([int(action), int(action)])
            move_count += 1

        # Determine game result
        # env.state is a list of two dictionaries, one for each player
        # Player 1 (index 0) reward: 1 if won, 0 if draw/loss, None if inactive
        # Player 2 (index 1) reward: 1 if won, 0 if draw/loss, None if inactive
        reward_p1 = env.state[0]['reward']
        reward_p2 = env.state[1]['reward']

        if reward_p1 == 1:
            value = 1.0 # Player 1 (our agent's first move perspective) won
        elif reward_p2 == 1:
             # Player 2 won. From Player 1's perspective, this is a loss.
             # Since the network always predicts the value from the current player's perspective,
             # and player perspectives alternate, we assign values accordingly later.
            value = -1.0 # Player 1 perspective loss
        else:
            value = 0.0 # Draw

        # Assign values and add examples
        for i in range(len(game_states)):
            # Value is from the perspective of the player whose turn it was
            # If i is even, it was Player 1's turn. If odd, Player 2's turn.
            # The stored value should reflect the *final* outcome from *that player's* perspective.
            player_perspective_value = value if i % 2 == 0 else -value
            examples.append((game_states[i], game_policies[i], player_perspective_value))

        return examples

    except Exception as e:
        logger.error(f"Error in self-play worker: {e}", exc_info=True)
        return None # Return None to indicate an error occurred


def train_network(model, optimizer, buffer, params, device):
    """Train the network using examples from the replay buffer on the specified device"""
    logger.info(f"Starting training on device: {device}")
    model.train() # Set model to training mode

    # Skip training if buffer doesn't have enough examples for a full batch
    if len(buffer) < params['batch_size']:
        logger.warning(f"Skipping training, buffer has only {len(buffer)} examples (Batch size: {params['batch_size']})")
        return 0, 0, 0

    total_loss = 0
    policy_loss_total = 0
    value_loss_total = 0

    # Calculate number of batches per epoch
    num_batches = len(buffer) // params['batch_size']
    if num_batches == 0:
         logger.warning(f"Skipping training, not enough samples for a single batch.")
         return 0, 0, 0

    # Use DataLoader for efficient batch loading (optional but good practice)
    # Note: DataLoader might be overkill if buffer.sample is already efficient
    # train_loader = DataLoader(buffer, batch_size=params['batch_size'], shuffle=True, num_workers=0) # num_workers=0 for simplicity here

    # Training loop with tqdm progress bar for epochs
    for epoch in tqdm(range(params['num_epochs']), desc="Training Epochs"):
        epoch_loss = 0
        epoch_policy_loss = 0
        epoch_value_loss = 0
        num_processed_batches = 0

        # Loop through batches (alternative: use buffer.sample in a loop)
        # for states, policies, values in tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['num_epochs']}", leave=False):
        for _ in range(num_batches): # Simple loop using buffer.sample
            # Get a batch of training data
            states, policies, values = buffer.sample(params['batch_size'])

            # Move data to the target device (GPU or CPU)
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            policy_logits, value_pred = model(states)

            # Calculate loss
            # Value loss: Mean Squared Error between predicted value and actual game outcome
            value_loss = F.mse_loss(value_pred, values)
            # Policy loss: Cross-Entropy between predicted policy logits and MCTS policy distribution
            # Note: MCTS policy is probabilities, cross_entropy expects target indices or probabilities.
            # If policies are probabilities (sum to 1), use KL divergence or adjust cross_entropy.
            # Assuming policies are target probabilities:
            policy_loss = -torch.sum(policies * F.log_softmax(policy_logits, dim=1), dim=1).mean()
            # Alternative if policies are action indices (less common in AlphaZero):
            # policy_loss = F.cross_entropy(policy_logits, policies.argmax(dim=1))

            loss = value_loss + policy_loss
            # logger.debug(f"Epoch {epoch+1}, Batch - value loss: {value_loss.item():.4f}, policy loss: {policy_loss.item():.4f}")

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Update metrics for the epoch
            epoch_loss += loss.item()
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            num_processed_batches += 1

        # Log epoch average losses
        avg_epoch_loss = epoch_loss / num_processed_batches
        avg_epoch_policy_loss = epoch_policy_loss / num_processed_batches
        avg_epoch_value_loss = epoch_value_loss / num_processed_batches
        logger.debug(f"Epoch {epoch+1} Avg Loss: {avg_epoch_loss:.4f}, Policy: {avg_epoch_policy_loss:.4f}, Value: {avg_epoch_value_loss:.4f}")

        total_loss += avg_epoch_loss
        policy_loss_total += avg_epoch_policy_loss
        value_loss_total += avg_epoch_value_loss

    # Return average losses over all epochs
    num_epochs_run = params['num_epochs']
    return total_loss / num_epochs_run, policy_loss_total / num_epochs_run, value_loss_total / num_epochs_run


def evaluate_model(current_model, previous_model, num_games, device, params):
    """Evaluate current model against previous version on the specified device"""
    logger.info(f"Starting evaluation: {num_games} games...")
    current_model.eval()
    previous_model.eval() # Ensure previous model is also in eval mode

    current_wins = 0
    previous_wins = 0
    draws = 0

    # Wrap game loop with tqdm
    for game_idx in tqdm(range(num_games), desc="Evaluation Games"):
        env = make("connectx", debug=False)
        env.reset()

        # Alternate which model goes first
        if game_idx % 2 == 0:
            model_p1, model_p2 = current_model, previous_model
            p1_is_current = True
        else:
            model_p1, model_p2 = previous_model, current_model
            p1_is_current = False

        while not env.done:
            current_player_idx = env.state[0]['observation']['mark'] - 1 # 0 for player 1, 1 for player 2
            active_model = model_p1 if current_player_idx == 0 else model_p2

            # Get state tensor and move to device
            state_tensor = cxnn.preprocess_input(env).to(device)

            with torch.no_grad(): # Disable gradient calculation for inference
                 # Use MCTS for evaluation for stronger play, or direct policy for speed
                 # Using direct policy head for faster evaluation:
                 policy_logits, _ = active_model(state_tensor.unsqueeze(0)) # Add batch dim
                 policy_probs = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

                 # Mask invalid moves
                 valid_actions = [c for c in range(env.configuration.columns) if env.observation.board[c] == 0]
                 masked_policy = np.zeros_like(policy_probs)
                 if valid_actions: # Check if list is not empty
                     masked_policy[valid_actions] = policy_probs[valid_actions]
                     if masked_policy.sum() > 1e-6: # Avoid division by zero if all valid moves have zero prob
                         masked_policy /= masked_policy.sum()
                     else: # If all valid moves have ~zero probability, choose uniformly
                         logger.warning("Evaluation: All valid moves have near-zero probability. Choosing uniformly.")
                         masked_policy[valid_actions] = 1.0 / len(valid_actions)

                     # Choose action greedily based on policy probabilities
                     action = np.argmax(masked_policy)
                 else:
                     # This case should ideally not happen if env.done is checked correctly
                     logger.error("Evaluation: No valid actions available but game not done?")
                     action = 0 # Default fallback action

                 # --- Alternatively, use MCTS for evaluation (slower but stronger) ---
                 # action, _ = mcts.select_action(
                 #     root_env=env,
                 #     model=active_model,
                 #     n_simulations=params.get('eval_n_simulations', 50), # Use fewer simulations for eval
                 #     c_puct=params['c_puct'],
                 #     temperature=0, # Greedy selection during evaluation
                 #     device=device
                 # )
                 # -----------------------------------------------------------------

            # Step the environment
            env.step([int(action), int(action)]) # Both players use their respective model's action

        # Determine winner
        reward_p1 = env.state[0]['reward']
        reward_p2 = env.state[1]['reward']

        if reward_p1 == 1: # Player 1 won
            if p1_is_current:
                current_wins += 1
            else:
                previous_wins += 1
        elif reward_p2 == 1: # Player 2 won
            if p1_is_current:
                previous_wins += 1 # P2 (previous model) won when P1 was current
            else:
                current_wins += 1 # P2 (current model) won when P1 was previous
        else: # Draw
            draws += 1

    # Avoid division by zero if num_games is 0
    if num_games == 0:
        win_rate = 0.0
    else:
        # Calculate win rate for the current model
        win_rate = (current_wins + 0.5 * draws) / num_games

    logger.info(f"Evaluation Result - Current Wins: {current_wins}, Previous Wins: {previous_wins}, Draws: {draws}, Win Rate: {win_rate:.4f}")
    return win_rate, current_wins, previous_wins, draws

def main():
    # --- Setup ---
    # Set multiprocessing start method (important for CUDA compatibility on some systems)
    # try:
    #     mp.set_start_method('spawn')
    # except RuntimeError:
    #     logger.warning("Multiprocessing start method already set.")
    #     pass

    # Determine device (GPU if available, else CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Initialize model and move it to the target device
    model = cxnn.ConnectXNet().to(device)
    optimizer = optim.AdamW( # Use AdamW for potentially better regularization
        model.parameters(),
        lr=TRAINING_PARAMS['learning_rate'],
        weight_decay=TRAINING_PARAMS['weight_decay']
    )
    replay_buffer = ReplayBuffer(max_size=TRAINING_PARAMS['buffer_size'])

    # Create directories for saving models
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/best", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    # --- Training Loop ---
    best_win_rate = 0.55 # Initial win rate threshold to beat the previous best
    previous_model_state_dict = None # Store state_dict instead of the whole model

    # Use tqdm for the main iteration loop
    for iteration in tqdm(range(TRAINING_PARAMS['num_iterations']), desc="Training Iterations"):
        iteration_start_time = time.time()
        logger.info(f"===== Starting Iteration {iteration+1}/{TRAINING_PARAMS['num_iterations']} =====")

        # --- Self-Play Phase (Parallel) ---
        self_play_start_time = time.time()
        logger.info(f"Starting self-play phase with {TRAINING_PARAMS['num_workers']} workers...")

        # Get the model's state dictionary on CPU to pass to workers
        # Use deepcopy to avoid potential issues with shared state if model is modified later
        current_model_state_dict = copy.deepcopy(model.state_dict())
        for key in current_model_state_dict:
             current_model_state_dict[key] = current_model_state_dict[key].cpu()


        # Prepare arguments for each worker
        worker_args = [(current_model_state_dict, TRAINING_PARAMS, str(device))
                       for _ in range(TRAINING_PARAMS['num_self_play_games'])]

        all_game_examples = []
        try:
            # Create a multiprocessing pool
            with mp.Pool(processes=TRAINING_PARAMS['num_workers']) as pool:
                # Use tqdm to show progress for collecting results from the pool
                results = list(tqdm(pool.imap_unordered(run_self_play_game, worker_args),
                                    total=TRAINING_PARAMS['num_self_play_games'],
                                    desc="Self-Play Games"))

                # Collect results and filter out None values (errors)
                for game_examples in results:
                    if game_examples is not None:
                        all_game_examples.extend(game_examples)
                    else:
                        logger.warning("A self-play worker returned an error (None).")

        except Exception as e:
             logger.error(f"Error during parallel self-play: {e}", exc_info=True)
             # Decide how to handle this - e.g., continue with collected data, stop, etc.
             # For now, log the error and continue if some data was collected.

        self_play_duration = time.time() - self_play_start_time
        logger.info(f"Self-play phase completed in {self_play_duration:.2f}s. Generated {len(all_game_examples)} examples.")

        # Add generated examples to the replay buffer
        if all_game_examples:
            add_start_time = time.time()
            for state, policy, value in all_game_examples:
                replay_buffer.add(state, policy, value) # Add method ensures state is on CPU
            add_duration = time.time() - add_start_time
            logger.info(f"Added examples to buffer (Size: {len(replay_buffer)}/{TRAINING_PARAMS['buffer_size']}) in {add_duration:.2f}s.")
        else:
             logger.warning("No examples generated from self-play in this iteration.")
             # Consider skipping training or evaluation if no new data is available
             # continue # Optional: skip to next iteration

        # --- Training Phase ---
        if len(replay_buffer) >= TRAINING_PARAMS['batch_size']:
            train_start_time = time.time()
            total_loss, policy_loss, value_loss = train_network(model, optimizer, replay_buffer, TRAINING_PARAMS, device)
            train_duration = time.time() - train_start_time
            logger.info(f"Training phase completed in {train_duration:.2f}s.")
            logger.info(f"Avg Training Losses - Total: {total_loss:.4f}, Policy: {policy_loss:.4f}, Value: {value_loss:.4f}")
        else:
            logger.info("Skipping training phase: not enough data in buffer.")


        # --- Save Checkpoint ---
        if (iteration + 1) % TRAINING_PARAMS['checkpoint_interval'] == 0 or iteration == TRAINING_PARAMS['num_iterations'] - 1:
            checkpoint_path = f"models/checkpoints/model_iter_{iteration+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        # --- Evaluation Phase ---
        if (iteration + 1) % TRAINING_PARAMS['eval_interval'] == 0 and iteration > 0:
            eval_start_time = time.time()
            logger.info("Starting evaluation against previous best model...")

            # Load the previous best model state for comparison
            # Ensure previous_model is created and moved to the correct device
            previous_model = cxnn.ConnectXNet().to(device)
            best_model_path = "models/best/best_model.pth"

            if os.path.exists(best_model_path):
                 try:
                    # Load the state dict, mapping location ensures it loads correctly regardless of where it was saved
                    previous_model_state_dict = torch.load(best_model_path, map_location=device)
                    previous_model.load_state_dict(previous_model_state_dict)
                    logger.info(f"Loaded previous best model from {best_model_path} for evaluation.")
                 except Exception as e:
                    logger.error(f"Failed to load previous best model: {e}. Skipping evaluation against best.", exc_info=True)
                    previous_model = None # Indicate failure to load
            else:
                 logger.warning("No previous best model found to evaluate against. Skipping evaluation.")
                 previous_model = None

            if previous_model:
                win_rate, _, _, _ = evaluate_model(
                    current_model=model,
                    previous_model=previous_model,
                    num_games=TRAINING_PARAMS['eval_games'],
                    device=device,
                    params=TRAINING_PARAMS # Pass params for potential MCTS settings in eval
                )

                # Save the current model as the best if it meets the win rate threshold
                if win_rate > best_win_rate:
                    logger.info(f"New best model found! Win rate: {win_rate:.4f} > {best_win_rate:.4f}")
                    best_win_rate = win_rate
                    best_path = "models/best/best_model.pth"
                    torch.save(model.state_dict(), best_path)
                    logger.info(f"Saved new best model to: {best_path}")
                    # Update the state dict for the next evaluation
                    # previous_model_state_dict = copy.deepcopy(model.state_dict())
                else:
                    logger.info(f"Current model did not surpass best model. Win rate: {win_rate:.4f}, Best rate: {best_win_rate:.4f}")
                    # Keep the old best model's state_dict for the next comparison
                    # model.load_state_dict(previous_model_state_dict) # Optionally revert if desired

            eval_duration = time.time() - eval_start_time
            logger.info(f"Evaluation phase completed in {eval_duration:.2f}s.")

        iteration_duration = time.time() - iteration_start_time
        logger.info(f"Iteration {iteration+1} completed in {iteration_duration:.2f}s")
        logger.info(f"Estimated time remaining: {iteration_duration * (TRAINING_PARAMS['num_iterations'] - (iteration + 1)):.2f}s")


    # --- Save Final Model ---
    final_path = "models/final_model.pth"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Training completed. Final model saved at: {final_path}")

if __name__ == "__main__":
    # Ensure multiprocessing works correctly when script is run directly
    # (Needed on Windows and sometimes macOS)
    # mp.freeze_support() # Uncomment if using 'fork' start method and encountering issues

    main()
