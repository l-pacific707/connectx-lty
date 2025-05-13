import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import torch.optim.lr_scheduler as lr_scheduler
import random
from collections import deque
import time
import pickle
import os
import logging # Standard logging
from kaggle_environments import make
from tqdm import tqdm
import copy
import math
import csv
import yaml

import MCTS_Connectx as mcts
import ConnectXNN as cxnn
# Import new logging setup functions
from logger_setup import listener_process_configure, worker_configurer, LOG_QUEUE_SENTINEL, get_logger
from rng_init import rng_worker_init


# Setup logger for the main process (primarily for console output before listener takes over)
# This logger instance will be replaced in workers by the queue-based setup.
logger = get_logger("train.py", "Play_and_Train.log") # Main process logger

with open("training_config.yaml", "r") as file:
    TRAINING_PARAMS = yaml.safe_load(file)

class ReplayBuffer(Dataset):
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        state_np, policy, value = self.buffer[idx]
        return state_np, policy, torch.tensor(value, dtype=torch.float32)

    def add(self, state_np, policy, value):
        policy_tensor = torch.from_numpy(policy).float()
        self.buffer.append((state_np, policy_tensor, value))

# Modified worker initialization to include log queue configuration
def combined_worker_init(base_seed_val, log_queue_val, worker_id_val, only_log_debug_for_worker_zero_val):
    """Initializes RNG and logging for worker processes."""
    # Initialize RNG
    rng_worker_init(base_seed_val) # base_seed_val should be unique per worker or carefully managed

    # Configure logging for this worker
    log_level_for_worker = logging.DEBUG if (only_log_debug_for_worker_zero_val and worker_id_val == 0) else logging.INFO
    worker_configurer(log_queue_val, worker_log_level=log_level_for_worker)
    
    # Optional: Get a logger instance to confirm setup if needed for debugging worker init
    # worker_init_logger = logging.getLogger(f"WorkerInit_{os.getpid()}")
    # worker_init_logger.info(f"Worker {os.getpid()} (ID {worker_id_val}) initialized with log level {log_level_for_worker}.")


def run_self_play_game(args):
    model_state_dict, params, device_str, worker_id = args
    device = torch.device(device_str)
    
    # Logging is now handled by the QueueHandler configured in combined_worker_init
    # The logger instance will automatically use it.
    game_logger = logging.getLogger("MCTS") # Or any other logger name used within MCTS/self-play

    # log_debug_messages: this was for specific file logging; now levels control verbosity to queue
    # If you still want only worker 0 to log DEBUG to the queue, that's handled by combined_worker_init.
    # game_logger.debug(f"Worker {worker_id} starting self-play game.") # Example debug log

    try:
        from rng_init import np_rng # Get the worker-specific RNG
        model = cxnn.ConnectXNet()
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()

        examples = []
        env = make("connectx", debug=False)
        current_state = env.reset() # Get initial state

        game_states = []
        game_policies = []

        temperature = params['temperature_init']
        move_count = 0

        while not env.done:
            # Adjust temperature and noise based on move count
            if move_count > params['temp_decay_steps'] and temperature > params['temperature_final']:
                temperature *= params['temperature_decay_factor']
            
            current_mcts_alpha = params['mcts_alpha_std']
            if move_count > params['noise_threshold']:
                current_mcts_alpha = 0.0 # Disable noise

            # The MCTS functions will use logging.getLogger("MCTS") which is now queue-based
            action, policy = mcts.select_action(
                root_env=env, # Pass the whole env
                model=model,
                n_simulations=params['n_simulations'],
                c_puct=params['c_puct'],
                c_fpu=params['c_fpu'],
                mcts_alpha=current_mcts_alpha,
                mcts_epsilon=params['mcts_epsilon'],
                np_rng=np_rng, # Use worker's RNG
                temperature=temperature,
                device=device,
                log_debug=(worker_id == 0 and params.get('enable_mcts_debug_for_worker_0', False)) # Example conditional MCTS debug
            )
            
            # Store state (as tensor on CPU) and policy (as numpy array)
            # Preprocess input directly here before storing if necessary
            # For replay buffer, we need the state that MCTS used for its policy.
            # The cxnn.preprocess_input(env) is appropriate here.
            state_tensor_for_buffer = cxnn.preprocess_input(env).cpu() # Ensure it's on CPU
            game_states.append(state_tensor_for_buffer)
            game_policies.append(policy)

            if action is None:
                game_logger.error(f"Worker {worker_id}: MCTS returned None action. Board: {env.state[0]['observation']['board']}")
                break

            # Determine whose turn it is to correctly apply the action
            # This logic might need adjustment based on your env.step expectations
            current_player_idx_for_step = env.state[0]['observation']['mark'] -1 # 0 for P1, 1 for P2
            step_actions = [None, None]
            step_actions[current_player_idx_for_step] = int(action)
            
            env.step(step_actions)
            move_count += 1

        # Determine game result from player 1's perspective
        reward_p1 = env.state[0]['reward']
        if reward_p1 == 1: value = 1.0    # Player 1 won
        elif reward_p1 == -1: value = -1.0 # Player 1 lost (Player 2 won)
        else: value = 0.0                 # Draw

        # Assign values and add examples
        for i in range(len(game_states)):
            # Value is from the perspective of the player whose turn it was for that state
            player_perspective_value = value if (env.state[0]['observation']['mark'] -1 == i % 2) else -value
            # Store state_tensor (CPU), policy (numpy), value (float)
            examples.append((game_states[i], game_policies[i], float(player_perspective_value)))
        
        # game_logger.debug(f"Worker {worker_id} finished self-play game. Generated {len(examples)} examples.")
        return examples

    except Exception as e:
        # Use the logger from this worker. It will go to the queue.
        worker_logger = logging.getLogger(f"SelfPlayWorker_{worker_id}")
        worker_logger.error(f"Error in self-play worker {worker_id}: {e}", exc_info=True)
        return None


def train_network(model, optimizer, scheduler, buffer, params, device, global_step_counter):
    train_logger = logging.getLogger("train.py") # Will use queue handler
    train_logger.info(f"Starting training on device: {device} with {len(buffer)} examples.")
    model.train()
    
    dataloader = DataLoader(
        buffer,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=params.get('dataloader_num_workers', 0), # Use config or default to 0
        pin_memory=torch.cuda.is_available()
    )
    
    total_loss_accum = 0.0
    policy_loss_accum = 0.0
    value_loss_accum = 0.0
    batches_processed = 0

    for epoch in range(params['num_epochs']):
        epoch_desc = f"Epoch {epoch+1}/{params['num_epochs']}"
        for states, policies, values in tqdm(dataloader, desc=epoch_desc, leave=False):
            # states are already tensors from ReplayBuffer, ensure they are on the correct device
            states = states.to(device, non_blocking=True) 
            policies = policies.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True).unsqueeze(1)
            
            # Ensure states have the correct shape [B, C, H, W]
            if states.dim() == 5 and states.shape[1] == 1: # [B, 1, C, H, W]
                states = states.squeeze(1)
            elif states.dim() != 4: # Should be [B, C, H, W]
                train_logger.error(f"Unexpected state tensor shape during training: {states.shape}")
                continue # Skip this batch

            optimizer.zero_grad()
            policy_logits, value_pred = model(states)
            value_loss = F.mse_loss(value_pred, values)
            policy_loss = -torch.sum(policies * F.log_softmax(policy_logits, dim=1), dim=1).mean()
            loss = value_loss + policy_loss

            loss.backward()
            optimizer.step()
            scheduler.step() # Call scheduler every step
            global_step_counter[0] += 1

            total_loss_accum += loss.item()
            policy_loss_accum += policy_loss.item()
            value_loss_accum += value_loss.item()
            batches_processed += 1
            
            if global_step_counter[0] % 100 == 0: # Log LR periodically
                current_lr = optimizer.param_groups[0]['lr']
                train_logger.debug(f"Global Step: {global_step_counter[0]}, Current LR: {current_lr:.7f}")


    if batches_processed > 0:
        avg_total_loss = total_loss_accum / batches_processed
        avg_policy_loss = policy_loss_accum / batches_processed
        avg_value_loss = value_loss_accum / batches_processed
        train_logger.info(f"Training Avg Losses - Total: {avg_total_loss:.4f}, Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f}")
        return avg_total_loss, avg_policy_loss, avg_value_loss
    else:
        train_logger.warning("No batches processed during training.")
        return 0.0, 0.0, 0.0


def run_single_evaluation_game_worker(args):
    current_model_state_dict, previous_model_state_dict, device_str, params, worker_id = args
    device = torch.device(device_str)
    eval_logger = logging.getLogger(f"EvalWorker_{worker_id}") # Worker-specific logger

    try:
        from rng_init import np_rng # Get worker's RNG
        current_model_worker = cxnn.ConnectXNet().to(device)
        current_model_worker.load_state_dict(current_model_state_dict)
        current_model_worker.eval()

        previous_model_worker = cxnn.ConnectXNet().to(device)
        previous_model_worker.load_state_dict(previous_model_state_dict)
        previous_model_worker.eval()
        
        env = make("connectx", debug=False)
        env.reset()

        if worker_id % 2 == 0:
            model_p1, model_p2 = current_model_worker, previous_model_worker
            p1_is_current = True
        else:
            model_p1, model_p2 = previous_model_worker, current_model_worker
            p1_is_current = False

        while not env.done:
            active_model = model_p1 if env.state[0]["status"] == "ACTIVE" else model_p2
            
            action, _ = mcts.select_action(
                root_env=env, model=active_model,
                n_simulations=params['n_simulations_eval'], c_puct=params['c_puct'],
                c_fpu=0.0, mcts_alpha=0.0, mcts_epsilon=0.0,
                np_rng=np_rng, temperature=0.0, device=device,
                log_debug=False # No detailed MCTS logs for eval
            )

            if action is None:
                eval_logger.error("MCTS returned None during evaluation.")
                return 0, p1_is_current 
            
            current_player_idx_for_step = env.state[0]['observation']['mark'] -1
            step_actions = [None, None]
            step_actions[current_player_idx_for_step] = int(action)
            env.step(step_actions)

        reward_p1 = env.state[0]['reward']
        if reward_p1 == 1: return 1, p1_is_current
        elif reward_p1 == -1: return -1, p1_is_current
        else: return 0, p1_is_current
    except Exception as e:
        eval_logger.error(f"Error in evaluation worker {worker_id}: {e}", exc_info=True)
        return 0, False # Default to draw, p1_is_current doesn't matter as much on error


def evaluate_model(current_model_sd, previous_model_sd, num_games, device_str, params, num_workers_eval):
    eval_main_logger = logging.getLogger("train.py") # Main process logger for this phase
    eval_main_logger.info(f"Starting parallel evaluation: {num_games} games with {num_workers_eval} workers...")
    
    current_wins = 0
    previous_wins = 0
    draws = 0

    current_model_cpu_sd = {k: v.cpu() for k, v in current_model_sd.items()}
    previous_model_cpu_sd = {k: v.cpu() for k, v in previous_model_sd.items()}

    worker_args_list = [
        (current_model_cpu_sd, previous_model_cpu_sd, device_str, params, game_idx)
        for game_idx in range(num_games)
    ]
    
    eval_base_seed = params['base_seed'] + 20000 # Different base seed for eval workers
    # The log_queue is passed to combined_worker_init which sets up QueueHandler
    log_queue_for_eval_pool = mp.Manager().Queue(-1) if params.get('separate_eval_log_queue', False) else mp.get_context().Queue(-1)


    # For evaluation, we might want to make debug logs from worker 0 less verbose or off
    only_log_debug_for_worker_zero_eval = params.get('eval_worker_0_debug', False)

    pool_init_args_eval = [(eval_base_seed + i, log_queue_for_eval_pool, i, only_log_debug_for_worker_zero_eval) for i in range(num_workers_eval)]


    # Initialize the pool with combined_worker_init for each worker
    # Need to ensure initargs are correctly passed if combined_worker_init is used for the pool
    # The `initializer` takes one function. `initargs` is a tuple of arguments for that function.
    # If each worker needs different initargs (like worker_id), maxtasksperchild=1 can re-initialize.
    # Or, pass worker_id as part of the task arguments if initializer cannot vary per worker.
    # For simplicity, we'll assume rng_worker_init is sufficient for distinct RNG,
    # and logging setup will be done via worker_configurer using the passed log_queue.
    # The `worker_id` is already part of `worker_args_list` for the task function.

    # Let's use a Pool initializer that takes the queue.
    eval_pool_init_args = (log_queue_for_eval_pool, logging.INFO) # Log queue and default level for eval workers

    with mp.Pool(processes=num_workers_eval, initializer=worker_configurer, initargs=eval_pool_init_args) as pool:
        results = list(tqdm(pool.imap_unordered(run_single_evaluation_game_worker, worker_args_list),
                            total=num_games, desc="Evaluation Games"))

    for game_result, p1_was_current_model in results:
        if game_result == 1:
            if p1_was_current_model: current_wins +=1
            else: previous_wins += 1
        elif game_result == -1:
            if p1_was_current_model: previous_wins += 1
            else: current_wins += 1
        else: draws +=1
            
    win_rate = (current_wins + 0.5 * draws) / max(1, num_games)
    eval_main_logger.info(f"Evaluation Result - Current Wins: {current_wins}, Previous (best) Wins: {previous_wins}, Draws: {draws}, Win Rate for Current Model: {win_rate:.4f}")
    return win_rate, current_wins, previous_wins, draws


def save_replay_buffer(buffer: ReplayBuffer, path: str):
    # Convert tensors in buffer to numpy for pickling if they are not already
    serializable_buffer_data = []
    for item in buffer.buffer:
        state_tensor, policy_tensor, value_float = item
        # Assuming state_tensor is already numpy or can be easily converted if it's tensor
        state_data = state_tensor.numpy() if isinstance(state_tensor, torch.Tensor) else state_tensor
        policy_data = policy_tensor.cpu().numpy() # Ensure policy is numpy
        serializable_buffer_data.append((state_data, policy_data, value_float))
    
    with open(path, 'wb') as f:
        pickle.dump(serializable_buffer_data, f)

def load_replay_buffer(path: str, buffer: ReplayBuffer):
    with open(path, 'rb') as f:
        data_loaded = pickle.load(f)
    for state_np, policy_np, value_float in data_loaded:
        # state_np is loaded as numpy, policy_np as numpy
        # ReplayBuffer.add will convert policy_np to tensor
        buffer.add(state_np, policy_np, value_float)


def save_loss_history(loss_history, filename="results/loss_history.csv"):
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
    try:
        if mp.get_start_method(allow_none=True) is None:
             mp.set_start_method('spawn', force=True) # force=True if already set by user
             logger.info("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        logger.warning("Multiprocessing start method already set or failed to set. Current method: %s", mp.get_start_method(allow_none=True))
    except AttributeError:
         logger.warning("mp.get_start_method not available. Skipping start method setting.")

    # --- Logging Setup ---
    log_queue = mp.Manager().Queue(-1) # Use a manager queue if passing between Process and Pool
    
    # Define configurations for loggers that will write to files via the listener
    log_file_configs = {
        "MCTS": "MCTS.log",
        "train.py": "Play_and_Train.log", # Logger for main training script
        "ConnectXNN": "ConnectXNN.log",    # Logger for NN module
        # Add other loggers if they need their own files
    }
    # Add worker-specific logger names if you want them to have separate files (handled by listener)
    for i in range(TRAINING_PARAMS['num_workers']):
        log_file_configs[f"SelfPlayWorker_{i}"] = f"worker_selfplay_{i}.log"
        log_file_configs[f"EvalWorker_{i}"] = f"worker_eval_{i}.log"


    listener = mp.Process(target=listener_process_configure, args=(log_queue, log_file_configs))
    listener.daemon = True # So it exits when main process exits
    listener.start()
    
    # Configure the main process's root logger to also use the queue for now
    # This way, initial logs from main also go through the listener
    # worker_configurer(log_queue, logging.INFO) # Configure main process logging to queue

    logger.info("Main process logging configured to use queue. Listener started.")


    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    model_train = cxnn.ConnectXNet().to(device)
    model_play = cxnn.ConnectXNet().to(device) # This will be the "best" model for playing
    
    best_model_path = "models/best/best_model.pth"
    last_model_path = "models/last_model.pth"

    # Load last model for training continuation
    if os.path.exists(last_model_path):
        try:
            model_train.load_state_dict(torch.load(last_model_path, map_location=device))
            logger.info(f"Loaded last training model from: {last_model_path}")
        except Exception as e:
            logger.error(f"Error loading last model: {e}. Starting fresh.", exc_info=True)
    else:
        logger.info("No last model found. Starting training from scratch.")

    # Load best model for self-play and as baseline for evaluation
    if os.path.exists(best_model_path):
        try:
            model_play.load_state_dict(torch.load(best_model_path, map_location=device))
            logger.info(f"Loaded best model for self-play from: {best_model_path}")
            # If model_train is fresh, initialize it with best_model weights
            if not os.path.exists(last_model_path): # Only if not continuing training
                 model_train.load_state_dict(model_play.state_dict())
                 logger.info("Initialized training model with best model weights.")
        except Exception as e:
            logger.error(f"Error loading best model: {e}. Model_play may be uninitialized or use model_train's state.", exc_info=True)
            model_play.load_state_dict(model_train.state_dict()) # Fallback
    else:
        logger.info("No best_model.pth found. model_play will use initial/last model_train weights.")
        model_play.load_state_dict(model_train.state_dict())


    optimizer = optim.AdamW(model_train.parameters(), lr=TRAINING_PARAMS['learning_rate'], weight_decay=TRAINING_PARAMS['weight_decay'])
    replay_buffer = ReplayBuffer(max_size=TRAINING_PARAMS['buffer_size'])

    if os.path.exists("models/replay_buffer.pkl"):
        try:
            load_replay_buffer("models/replay_buffer.pkl", replay_buffer)
            logger.info(f"Loaded replay buffer. Size: {len(replay_buffer)}")
        except Exception as e:
            logger.error(f"Error loading replay buffer: {e}. Starting with empty buffer.", exc_info=True)

    estimated_batches_per_epoch = max(1, TRAINING_PARAMS['buffer_size'] // TRAINING_PARAMS['batch_size'])
    total_steps = TRAINING_PARAMS['num_iterations'] * TRAINING_PARAMS['num_epochs'] * estimated_batches_per_epoch
    warmup_steps = int(TRAINING_PARAMS['warmup_steps_ratio'] * total_steps)
    cosine_steps = max(1, total_steps - warmup_steps)
    initial_lr = TRAINING_PARAMS['learning_rate']
    min_lr = TRAINING_PARAMS['learning_rate'] * 0.05

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, cosine_steps))
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale = (1.0 - min_lr / initial_lr) * cosine_decay + (min_lr / initial_lr)
            return scale
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    global_step_counter = [0]
    loss_history = []
    os.makedirs("models/best", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    win_rate_threshold = 0.55
    base_seed = TRAINING_PARAMS["base_seed"]
    all_start_time = time.time()

    for iteration in range(TRAINING_PARAMS['num_iterations']):
        iteration_start_time = time.time()
        iter_num = iteration + 1
        logger.info(f"===== Starting Iteration {iter_num}/{TRAINING_PARAMS['num_iterations']} =====")
        
        current_model_state_dict_play = {k: v.cpu() for k, v in model_play.state_dict().items()}
        
        worker_args_self_play = [(current_model_state_dict_play, TRAINING_PARAMS, str(device), i)
                                 for i in range(TRAINING_PARAMS['num_self_play_games'])]

        all_game_examples = []
        
        # Determine if MCTS debug logs should be enabled for worker 0 during self-play
        # This can be a parameter in training_config.yaml if needed
        enable_mcts_debug_worker_0 = TRAINING_PARAMS.get('enable_mcts_debug_for_worker_0', False)
        
        # Prepare initargs for the self-play pool
        # Each worker gets a unique seed and the log_queue
        # Also pass the worker_id for the combined_worker_init to know its ID
        pool_initargs_self_play = [(base_seed + iter_num * TRAINING_PARAMS['num_workers'] + i, log_queue, i, enable_mcts_debug_worker_0) 
                                   for i in range(TRAINING_PARAMS['num_workers'])]


        try:
            # Use a context manager for the Pool
            # Pass a unique initargs tuple for each worker if maxtasksperchild=1
            # Or, ensure the initializer can handle being called once per worker with a shared base_seed/queue
            # For combined_worker_init, we pass worker_id to it, so it must be unique per worker.
            # If num_workers > number of items in pool_initargs_self_play, it will cycle.
            # The initializer is called ONCE per child process.
            # So, the initargs should be a single tuple if all workers get the same init args.
            # If they need different init args (like a worker_id), that's more complex with initializer.
            # Let's simplify: worker_configurer is called in initargs, and it doesn't need worker_id.
            # rng_worker_init in combined_worker_init uses os.getpid() for uniqueness.
            
            # Corrected initargs for the pool: a single tuple of args for the initializer function
            # The initializer `combined_worker_init` will be called for each worker process.
            # It needs the base_seed and the log_queue. Worker ID is implicit (os.getpid()).
            # The `only_log_debug_for_worker_zero_val` needs to be decided if it's based on a fixed worker 0
            # or the first worker started by the pool. For simplicity now, let's assume worker 0 in the *task* args.

            # The initializer needs: base_seed for RNG, log_queue, and a flag for worker 0 debug level
            # For `combined_worker_init(base_seed_val, log_queue_val, worker_id_val, only_log_debug_for_worker_zero_val)`:
            # We cannot directly pass worker_id_val to the pool's *initializer* in a way that it differs per worker.
            # So, `combined_worker_init` should not rely on a passed `worker_id_val`.
            # Let `rng_worker_init` handle unique seeding, and `worker_configurer` handle queue setup.
            # The `log_debug_messages` for the task `run_self_play_game` will use its `worker_id` arg.

            # Simpler Pool init:
            self_play_pool_init_args = (base_seed + iter_num, log_queue, enable_mcts_debug_worker_0) 
            # This implies `combined_worker_init` needs to be adjusted or we use two separate initializers.
            # Let's stick to one initializer if possible.
            # `rng_worker_init` will use `base_seed_val + os.getpid()`
            # `worker_configurer` will use `log_queue_val` and a default log level.
            # The `log_debug_messages` in `run_self_play_game` will control if *MCTS* logs debug details.

            with mp.Pool(processes=TRAINING_PARAMS['num_workers'],
                         initializer=worker_configurer, # Simpler: just configure logging
                         initargs=(log_queue, logging.DEBUG if enable_mcts_debug_worker_0 else logging.INFO) # Pass queue and default level
                        ) as pool:
                # rng_worker_init should be called inside the task or if we ensure per-worker seeding.
                # If worker_configurer is the initializer, RNG must be handled within the task or by a different initializer.
                # Let's assume rng_worker_init is called at the start of `run_self_play_game` for now.
                # This is not ideal for Pool's initializer.
                # A better way for Pool:
                # def pool_initializer_func(log_q, b_seed, debug_w0):
                #     worker_configurer(log_q, logging.DEBUG if debug_w0 else logging.INFO)
                #     rng_worker_init(b_seed) # This will be called with the same b_seed for all workers in this pool
                                            # if b_seed is not made unique per worker.
                                            # rng_worker_init uses os.getpid() so it's fine.

                def self_play_initializer(log_q, b_seed_iter, enable_debug_for_mcts_w0):
                    worker_configurer(log_q, logging.DEBUG) # Log everything to queue, MCTS internal debug controlled by task
                    rng_worker_init(b_seed_iter + os.getpid()) # Ensure unique seed per worker

                current_self_play_seed = base_seed + iter_num * 100 # Base seed for this iteration's self-play workers

                with mp.Pool(processes=TRAINING_PARAMS['num_workers'],
                             initializer=self_play_initializer,
                             initargs=(log_queue, current_self_play_seed, enable_mcts_debug_worker_0)
                            ) as pool:
                    results = list(tqdm(pool.imap_unordered(run_self_play_game, worker_args_self_play),
                                        total=TRAINING_PARAMS['num_self_play_games'],
                                        desc=f"Iter {iter_num} Self-Play"))
                    for game_examples in results:
                        if game_examples:
                            all_game_examples.extend(game_examples)
        except Exception as e:
             logger.error(f"Error during parallel self-play: {e}", exc_info=True)

        logger.info(f"Self-play generated {len(all_game_examples)} examples.")
        if all_game_examples:
            for state, policy, value in all_game_examples:
                replay_buffer.add(state.numpy(), policy, value) # Ensure state is numpy if tensor before
            logger.info(f"Buffer size: {len(replay_buffer)}")

        if len(replay_buffer) >= TRAINING_PARAMS['batch_size']:
            avg_iter_losses = train_network(model_train, optimizer, scheduler, replay_buffer, TRAINING_PARAMS, device, global_step_counter)
            loss_history.append(avg_iter_losses)
        else:
            logger.info("Skipping training: not enough data.")
            loss_history.append((0,0,0))


        if iter_num % TRAINING_PARAMS['checkpoint_interval'] == 0 or iter_num == TRAINING_PARAMS['num_iterations']:
            torch.save(model_train.state_dict(), f"models/checkpoints/model_iter_{iter_num}.pth")
            torch.save(model_train.state_dict(), last_model_path)
            save_replay_buffer(replay_buffer, "models/replay_buffer.pkl")
            logger.info(f"Saved checkpoint and replay buffer at iteration {iter_num}.")

        if iter_num % TRAINING_PARAMS['eval_interval'] == 0 and iter_num > 0:
            win_rate, _, _, _ = evaluate_model(
                model_train.state_dict(), model_play.state_dict(),
                TRAINING_PARAMS['eval_games'], str(device), TRAINING_PARAMS, TRAINING_PARAMS['num_workers']
            )
            if win_rate > win_rate_threshold:
                logger.info(f"New best model! Win rate: {win_rate:.4f}")
                torch.save(model_train.state_dict(), best_model_path)
                model_play.load_state_dict(model_train.state_dict())
                # Optionally prune buffer: replay_buffer.buffer = deque(list(replay_buffer.buffer)[-int(len(replay_buffer.buffer)*0.8):])
            else:
                logger.info(f"Current model win rate {win_rate:.4f} did not exceed threshold {win_rate_threshold}.")
        
        logger.info(f"Iteration {iter_num} took {time.time() - iteration_start_time:.2f}s.")


    save_loss_history(loss_history)
    logger.info(f"Total training time: {time.time() - all_start_time:.2f}s")

    # Signal listener to stop and wait for it
    logger.info("Sending sentinel to log listener...")
    log_queue.put(LOG_QUEUE_SENTINEL)
    log_queue.close() # Close the queue from this end
    log_queue.join_thread() # Wait for all items to be flushed
    
    # Timeout for listener join, as it might be stuck if sentinel not processed.
    listener.join(timeout=5) 
    if listener.is_alive():
        logger.warning("Log listener did not terminate gracefully. Forcing termination.")
        listener.terminate() # Force if needed
    else:
        logger.info("Log listener terminated.")


if __name__ == "__main__":
    main()