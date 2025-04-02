import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from collections import deque
import time
import os
import logging
from kaggle_environments import make

# Import your existing modules
import MCTS_Connectx as mcts
import ConectXNN as cxnn
import logger_setup

# Setup logger
logger = logging.getLogger("AlphaZeroTraining")
file_handler = logging.FileHandler('AlphaZeroTraining.log')
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)

# Training Parameters
TRAINING_PARAMS = {
    # MCTS parameters
    'n_simulations': 10,     # Number of MCTS simulations per move
    'c_puct': 1.0,            # Exploration constant for MCTS
    
    # Self-play parameters
    'num_self_play_games': 100,  # Number of self-play games per iteration
    'temperature_init': 1.0,     # Initial temperature for action selection
    'temperature_decay_factor' : 0.95, # Decay factor for temperature
    'temperature_final': 0.1,    # Final temperature after temp_decay_steps
    'temp_decay_steps': 10,      # Number of moves before temperature decay
    
    # Training parameters
    'batch_size': 128,         # Training batch size
    'buffer_size': 10000,      # Maximum size of replay buffer
    'num_epochs': 10,          # Epochs per training iteration
    'learning_rate': 0.001,    # Learning rate
    'weight_decay': 1e-4,      # L2 regularization parameter
    
    # Checkpoint parameters
    'checkpoint_interval': 20, # Save model every n iterations
    'eval_interval': 10,       # Evaluate against previous version every n iterations
    'eval_games': 10,          # Number of evaluation games
    
    # Training iterations
    'num_iterations': 50       # Total number of iterations
}

class ReplayBuffer(Dataset):
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
        
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        state, policy, value = self.buffer[idx]
        return state, policy, value
    
    def add(self, state, policy, value):
        self.buffer.append((state, policy, value))
    
    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        samples = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*samples)
        return torch.cat(states, dim=0), torch.tensor(policies), torch.tensor(values).unsqueeze(1)

def self_play(model, params):
    """Execute one game of self-play to generate training data"""
    model.eval()  # Set model to evaluation mode
    examples = []
    env = make("connectx", debug=True)
    observation = env.reset()
    
    # Track states and policies for later assigning game outcome
    game_states = []
    game_policies = []
    
    temperature = params['temperature_init']
    move_count = 0
    
    while not env.done:
        logger.debug(f"current move_count : {move_count}")
        # Decay temperature after certain number of moves
        if move_count >= params['temp_decay_steps']:
            if temperature > params['temperature_final']:
                temperature *= params['temperature_decay_factor']
        
        # Get state representation
        state_tensor = cxnn.preprocess_input(env)
        
        # Get action and policy from MCTS
        action, policy = mcts.select_action(
            root_env=env,
            model=model,
            n_simulations=params['n_simulations'],
            c_puct=params['c_puct'],
            temperature=temperature
        )
        
        # Save state and policy
        game_states.append(state_tensor)
        game_policies.append(policy)
        
        # Execute action
        env.step([action, action])  # The opponent's action is determined by the environment
        move_count += 1
    
    # Determine game result
    if env.state[0]['status'] == 'DONE' and env.state[0]['reward'] == 1:
        # First player (our agent) won
        value = 1.0
    elif env.state[1]['status'] == 'DONE' and env.state[1]['reward'] == 1:
        # Second player (also our agent) won
        value = -1.0
    else:
        # Draw
        value = 0.0
    
    # Assign values and add examples
    for i in range(len(game_states)):
        # Alternate value for player 2's perspective
        player_perspective_value = value if i % 2 == 0 else -value
        examples.append((game_states[i], game_policies[i], player_perspective_value))
    
    return examples

def train_network(model, optimizer, buffer, params):
    """Train the network using examples from the replay buffer"""
    model.train()  # Set model to training mode
    
    # Skip training if buffer doesn't have enough examples
    if len(buffer) < params['batch_size']:
        logger.info(f"Skipping training, buffer has only {len(buffer)} examples")
        return 0, 0, 0
    
    total_loss = 0
    policy_loss_total = 0
    value_loss_total = 0
    num_batches = 0
    
    for _ in range(params['num_epochs']):
        # Get a batch of training data
        states, policies, values = buffer.sample(params['batch_size'])
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        policy_logits, value_pred = model(states)
        
        # Calculate loss
        value_loss = F.mse_loss(value_pred, values.float())
        policy_loss = -torch.mean(torch.sum(policies * F.log_softmax(policy_logits, dim=1), dim=1))
        loss = value_loss + policy_loss
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        policy_loss_total += policy_loss.item()
        value_loss_total += value_loss.item()
        num_batches += 1
    
    # Return average losses
    return total_loss / num_batches, policy_loss_total / num_batches, value_loss_total / num_batches

def evaluate_model(current_model, previous_model, num_games=10):
    """Evaluate current model against previous version"""
    current_wins = 0
    previous_wins = 0
    draws = 0
    
    for game_idx in range(num_games):
        env = make("connectx", debug=True)
        env.reset()
        
        # Alternate which model goes first
        if game_idx % 2 == 0:
            first_model, second_model = current_model, previous_model
        else:
            first_model, second_model = previous_model, current_model
        
        while not env.done:
            if env.state[0]['status'] == 'ACTIVE':  # First player's turn
                model = first_model
            else:  # Second player's turn
                model = second_model
            
            action, _ = cxnn.select_action(model, env)
            
            env.step([action, action])
        
        # Determine winner
        if env.state[0]['reward'] == 1:  # First player won
            if game_idx % 2 == 0:
                current_wins += 1
            else:
                previous_wins += 1
        elif env.state[1]['reward'] == 1:  # Second player won
            if game_idx % 2 == 0:
                previous_wins += 1
            else:
                current_wins += 1
        else:  # Draw
            draws += 1
    
    win_rate = (current_wins + 0.5 * draws) / num_games
    return win_rate, current_wins, previous_wins, draws

def main():
    # Initialize model, optimizer, and replay buffer
    model = cxnn.ConnectXNet()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=TRAINING_PARAMS['learning_rate'], 
        weight_decay=TRAINING_PARAMS['weight_decay']
    )
    replay_buffer = ReplayBuffer(max_size=TRAINING_PARAMS['buffer_size'])
    
    # Create directories for saving models
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/best", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    
    # Training iterations
    best_win_rate = 0.5  # Initial win rate threshold
    previous_model = None
    
    for iteration in range(TRAINING_PARAMS['num_iterations']):
        start_time = time.time()
        logger.info(f"Starting iteration {iteration+1}/{TRAINING_PARAMS['num_iterations']}")
        
        # Self-play phase
        examples = []
        for game in range(TRAINING_PARAMS['num_self_play_games']):
            game_examples = self_play(model, TRAINING_PARAMS)
            examples.extend(game_examples)
            logger.debug(f"Completed self-play game {game+1}/{TRAINING_PARAMS['num_self_play_games']}")
        
        # Add examples to replay buffer
        for state, policy, value in examples:
            replay_buffer.add(state, policy, value)
        
        logger.info(f"Buffer size: {len(replay_buffer)}")
        
        # Training phase
        total_loss, policy_loss, value_loss = train_network(model, optimizer, replay_buffer, TRAINING_PARAMS)
        logger.info(f"Training losses - Total: {total_loss:.4f}, Policy: {policy_loss:.4f}, Value: {value_loss:.4f}")
        
        # Save checkpoint
        if (iteration + 1) % TRAINING_PARAMS['checkpoint_interval'] == 0:
            checkpoint_path = f"models/checkpoints/model_iter_{iteration+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Evaluate against previous version
        if (iteration + 1) % TRAINING_PARAMS['eval_interval'] == 0 and iteration > 0:
            # Create a copy of the current model for evaluation
            if previous_model is None:
                previous_model = cxnn.ConnectXNet()
                # Load an older version of the model
                prev_checkpoint = f"models/checkpoints/model_iter_{iteration+1-TRAINING_PARAMS['eval_interval']}.pth"
                previous_model.load_state_dict(torch.load(prev_checkpoint))
            
            win_rate, current_wins, previous_wins, draws = evaluate_model(
                model, previous_model, TRAINING_PARAMS['eval_games']
            )
            
            logger.info(f"Evaluation - Win rate: {win_rate:.4f}, Wins: {current_wins}, Losses: {previous_wins}, Draws: {draws}")
            
            # Save as best model if improved
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_path = "models/best/best_model.pth"
                torch.save(model.state_dict(), best_path)
                logger.info(f"New best model with win rate {win_rate:.4f}")
        
        iteration_time = time.time() - start_time
        logger.info(f"Iteration {iteration+1} completed in {iteration_time:.2f}s")
    
    # Save final model
    final_path = "models/final_model.pth"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Training completed. Final model saved at: {final_path}")

if __name__ == "__main__":
    main()