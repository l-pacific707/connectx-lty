# MCTS_ConnectX.py
"""# MCTS 기반 AlphaZero-style 노드 구조 및 탐색 설계
Based on AlphaZero MCTS node structure and search design.
"""
from logger_setup import get_logger # Assuming you have this setup
import math
import numpy as np
from collections import defaultdict
import copy
import torch

# Assuming ConnectXNN is in the same directory or PYTHONPATH
import ConnectXNN as cxnn # <<< Corrected import name


logger = get_logger("MCTS","MCTS.log")

# Device handling (will be determined in train.py and passed if needed,
# but inference within MCTS primarily uses the model passed in)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define device in the main script

class MCTSNode:
    def __init__(self, state, parent=None, prior=0.0):
        self.state = state # The environment state will be used
        self.parent = parent
        self.children = {}
        # Determine current player based on parent or initial state
        if parent is not None:
            # Player alternates: 1 -> 2, 2 -> 1
            temp_players = {1: 2, 2: 1}
            self.current_player = temp_players.get(parent.current_player)
            if self.current_player is None:
                 logger.error(f"Parent node had invalid current_player: {parent.current_player}")
                 # Attempt recovery or raise error
                 self.current_player = self.find_current_player() # Fallback
        else:
            self.current_player = self.find_current_player()

        self.N = 0      # Visit count
        self.W = 0.0    # Total action value
        self.Q = 0.0    # Mean action value
        self.P = prior  # Prior probability from network

    def is_expanded(self):
        return len(self.children) > 0

    def expand(self, action_priors):
        """Expand the node by creating children for valid actions."""
        for action, prob in action_priors:
            if action not in self.children:
                # Child state is initially None; it will be set when the child is selected/simulated
                self.children[action] = MCTSNode(state=None, parent=self, prior=prob)

    def select_child(self, c_puct):
        """Select the child with the highest UCB score."""
        best_score = -float('inf')
        best_action = None
        best_child = None

        sqrt_total_N = math.sqrt(self.N)

        for action, child in self.children.items():
            # UCB calculation: Q + U
            # U = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            u = c_puct * child.P * sqrt_total_N / (1 + child.N)
            score = child.Q + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        if best_child is None:
            logger.warning(f"No best child found for node. State:\n{convert_board_to_2D(self.state) if self.state else 'None'}")
            # Handle cases where no valid child might be selectable (e.g., only invalid actions left?)
            # As a fallback, maybe select randomly or the first child?
            if self.children:
                best_action = list(self.children.keys())[0]
                best_child = self.children[best_action]
            else:
                return None, None # Cannot select if no children

        return best_action, best_child

    def backup(self, value):
        """Backup the value estimate through the tree path."""
        node = self
        current_value = value
        while node is not None:
            node.N += 1
            node.W += current_value
            node.Q = node.W / node.N
            # The value should be negated for the parent, as it represents the opponent's perspective
            current_value = -current_value
            node = node.parent

    def find_current_player(self) -> int:
        """Determine the current player from the environment state."""
        if self.state is None:
             logger.error("Cannot determine current player without a state.")
             # This should ideally not happen for the root node
             return 1 # Default guess or raise error

        # Check Kaggle environment state structure
        if self.state.state[0]["status"] == "ACTIVE":
            return 1
        elif self.state.state[1]["status"] == "ACTIVE":
            return 2
        else: # Game is DONE or INVALID
            logger.debug(f"Current state is terminal. State: {self.state.state}")
            # If terminal, the 'current' player doesn't really matter for selection,
            # but for consistency, determine who *would* have played.
            # This depends on whose turn it was when the game ended.
            # Let's assume the player who just made the move caused the end state.
            # The opponent of that player would be 'next'.
            # Reverting to parent's player logic might be simpler if available.
            if self.parent:
                temp_players = {1: 2, 2: 1}
                opponent = temp_players.get(self.parent.current_player)
                if opponent:
                    return opponent
                else:
                    logger.error("Parent node had invalid current_player value.")
                    # Fallback if parent info is wrong or unavailable
                    # Count pieces? More robust but slower.
                    board = np.array(self.state.state[0]["observation"]["board"])
                    p1_pieces = np.sum(board == 1)
                    p2_pieces = np.sum(board == 2)
                    return 2 if p1_pieces > p2_pieces else 1 # Guess based on piece count
            else:
                 logger.error("Cannot determine player for terminal root node without parent.")
                 return 1 # Default guess


def simulate_env(env, action):
    """Creates a deep copy of the environment and performs a step."""
    env_copy = copy.deepcopy(env)
    # Kaggle env step takes a list [player1_action, player2_action]
    # During simulation, we only care about the action of the current player
    # The environment handles the turn internally.
    # We can pass None or the same action for the inactive player.
    current_player_idx = 0 if env_copy.state[0]["status"] == "ACTIVE" else 1
    actions = [None, None]
    actions[current_player_idx] = action
    env_copy.step(actions)
    return env_copy

def is_terminal(env):
    """Check if the game has ended."""
    return env.done # Kaggle env provides a .done attribute

def get_valid_actions(env):
    """Get a list of valid actions (columns where a piece can be dropped)."""
    board = np.array(env.state[0]["observation"]["board"])
    columns = env.configuration.columns
    # A column is valid if the top cell (index c) is empty (0)
    return [c for c in range(columns) if board[c] == 0]

def convert_board_to_2D(env):
    """Helper to visualize the board from the 1D list."""
    if not env or not env.state: return "Invalid env for board conversion"
    board = np.array(env.state[0]["observation"]["board"])
    rows = env.configuration.rows
    columns = env.configuration.columns
    return board.reshape((rows, columns))


def get_game_result(env, perspective_player):
    """
    Determine the game result (+1 win, -1 loss, 0 draw) from the perspective
    of the given player.
    """
    if not env.done:
        logger.error("Requesting game result for a non-terminal state.")
        # raise ValueError("Game is not finished.")
        return 0 # Or handle as appropriate

    player_index = perspective_player - 1 # 0 for player 1, 1 for player 2

    # Check status and reward from the perspective_player's state
    status = env.state[player_index]["status"]
    reward = env.state[player_index]["reward"]

    if status == "DONE":
        if reward == 1: # Won
            return 1.0
        elif reward == 0: # Draw (Kaggle ConnectX uses 0.5 for draw reward?) Check env spec. Assuming 0 for now.
            # Check opponent's reward. If opponent also has 0 -> Draw.
            opponent_index = 1 - player_index
            opponent_reward = env.state[opponent_index]["reward"]
            if opponent_reward == 0 or opponent_reward == 0.5: # Adjust based on actual draw reward
                 return 0.0
            else: # This case should ideally not happen if one player gets 0 reward
                 logger.warning(f"Unexpected reward structure at game end. P{perspective_player} reward: {reward}, Opponent reward: {opponent_reward}")
                 return 0.0 # Default to draw
        elif reward < 0: # Lost (Kaggle uses -1 for loss?) Check env spec.
             return -1.0
        else: # Other reward values? e.g. Draw reward 0.5
             # Assuming the reward reflects the outcome directly for the player
             if reward > 0 and reward < 1: # Treat partial rewards as draw for simplicity, or adjust logic
                  return 0.0
             else:
                  logger.warning(f"Unexpected reward value {reward} for player {perspective_player} at game end.")
                  return 0.0 # Default
    elif status == "INVALID":
        logger.warning(f"Game ended with INVALID status for player {perspective_player}.")
        # Penalize the player who caused the invalid move?
        # Let's assume an invalid move is a loss for that player.
        return -1.0
    elif status == "ERROR":
         logger.error("Game ended with ERROR status.")
         return 0.0 # Treat as draw or handle differently
    else:
        logger.error(f"Game is done but status is unexpected: {status}")
        return 0.0

def make_tree(root_env, model, n_simulations, c_puct, device):
    """Perform MCTS simulations starting from the root_env."""
    root_node = MCTSNode(state=root_env)
    logger.debug(f"Start MCTS. Root state:\n{convert_board_to_2D(root_env)}\nCurrent player: {root_node.current_player}")

    # Initial evaluation and expansion of the root node
    with torch.no_grad(): # <<< Ensure no gradients during MCTS inference
        input_tensor = cxnn.preprocess_input(root_env).to(device) # <<< Move input to device
        p_logits, v = model(input_tensor)
        # Ensure output is detached and moved to CPU for numpy conversion
        p = torch.softmax(p_logits, dim=-1).detach().cpu().numpy().flatten()
        value_estimate = v.item() # Get scalar value

    valid_actions = get_valid_actions(root_env)
    if not valid_actions:
         logger.warning("No valid actions from root state. Cannot perform MCTS.")
         # This might happen if the game is already over when make_tree is called.
         return root_node # Return the unexpanded root

    action_priors = [(a, p[a]) for a in valid_actions]
    # Filter priors for valid actions and re-normalize? AlphaZero does not re-normalize.
    root_node.expand(action_priors)
    # Backup the initial value estimate. Should this be done?
    # AlphaZero backups the *outcome* (or network eval) from the LEAF node.
    # Let's skip root backup here and rely on simulation backups.
    # root_node.backup(value_estimate) # <<< Reconsider this line's necessity. Usually backup starts from leaf eval/outcome.
    logger.debug(f"Root node initial value estimate: {value_estimate:.4f}")


    for sim in range(n_simulations):
        node = root_node
        simulation_env = root_env # Start simulation from the original root state

        # --- Selection Phase ---
        while node.is_expanded():
            action, node = node.select_child(c_puct)
            if action is None or node is None:
                 logger.warning(f"Selection failed at simulation {sim+1}. Stopping this sim.")
                 break # Stop this simulation if selection fails

            logger.debug(f"Sim {sim+1}: Selected action {action}")
            # Simulate the action in the environment
            simulation_env = simulate_env(simulation_env, action)

            # If the node hasn't been assigned a state yet (it was created during expansion), assign it now.
            if node.state is None:
                node.state = simulation_env

            if is_terminal(simulation_env):
                logger.debug(f"Sim {sim+1}: Reached terminal state during selection.")
                break # Move to backup phase

        if action is None : continue # Skip if selection failed early

        # --- End of Selection (Reached a leaf or terminal state) ---

        value = 0.0 # Value to backup
        if is_terminal(simulation_env):
            # Game ended, get the actual outcome
            # The value should be from the perspective of the player whose turn it *would* be
            # at the state *before* the terminal state was reached (i.e., node.parent.current_player).
            # Or, get result relative to node.current_player and negate during backup.
            perspective_player = node.current_player # Player who would play at this terminal state
            game_outcome = get_game_result(simulation_env, perspective_player)
            value = game_outcome
            logger.debug(f"Sim {sim+1}: Terminal state reached. Outcome for P{perspective_player}: {value}")
        else:
            # --- Expansion & Evaluation Phase (Reached a leaf node) ---
            # Node state should be set now (either root or set during selection)
            if node.state is None:
                 logger.error(f"Sim {sim+1}: Reached non-terminal leaf node with no state. Logic error?")
                 node.state = simulation_env # Attempt recovery
                 # continue # Skip backup for this sim?

            logger.debug(f"Sim {sim+1}: Reached leaf node. Evaluating.")
            with torch.no_grad(): # <<< Ensure no gradients
                input_tensor = cxnn.preprocess_input(node.state).to(device) # <<< Move input to device
                p_logits, v_leaf = model(input_tensor)
                p_leaf = torch.softmax(p_logits, dim=-1).detach().cpu().numpy().flatten()
                value = v_leaf.item() # Use network's value estimate

            valid_actions_leaf = get_valid_actions(node.state)
            if valid_actions_leaf:
                leaf_priors = [(a, p_leaf[a]) for a in valid_actions_leaf]
                node.expand(leaf_priors)
                logger.debug(f"Sim {sim+1}: Leaf node expanded with {len(valid_actions_leaf)} actions.")
            else:
                 logger.debug(f"Sim {sim+1}: Leaf node has no valid actions (likely terminal state missed?).")
                 # If it should have been terminal, calculate true reward?
                 if not is_terminal(node.state):
                      logger.warning("Non-terminal leaf node has no valid actions!")
                 # Use the evaluated value anyway? Or terminal reward?
                 perspective_player = node.current_player
                 game_outcome = get_game_result(node.state, perspective_player)
                 value = game_outcome # Override network value with true outcome if possible


        # --- Backup Phase ---
        logger.debug(f"Sim {sim+1}: Backing up value: {value:.4f}")
        node.backup(value) # Backup starts from the expanded/terminal node

    return root_node

def create_pi(root_node, num_actions, temperature=1.0):
    """
    Create the policy vector pi based on node visit counts N.
    pi(a|s) = N(s,a)^(1/temp) / sum_b(N(s,b)^(1/temp))
    """
    pi = np.zeros(num_actions, dtype=np.float32)
    visit_counts = np.zeros(num_actions, dtype=np.float32)

    # Ensure children exist before iterating
    if not root_node.children:
        logger.warning("Root node has no children, cannot create policy pi. Returning uniform.")
        # This might happen if MCTS couldn't run (e.g., no valid actions initially)
        # Return a uniform distribution over valid actions as a fallback.
        # Need the environment state to know valid actions. This function lacks it.
        # Returning uniform over ALL actions might lead to invalid moves.
        # Let the caller handle this based on the context.
        return pi # Return zeros, caller must handle

    for action, child in root_node.children.items():
        if action < num_actions: # Ensure action index is within bounds
            visit_counts[action] = child.N
        else:
            logger.warning(f"Action {action} out of bounds (num_actions={num_actions}) in create_pi.")


    if root_node.N == 0 : # If root was never visited (e.g. MCTS failed)
         logger.warning("Root node visit count is zero in create_pi. Returning uniform.")
         # Again, need valid actions. Returning zeros.
         return pi


    if temperature == 0:
        # Deterministic selection: choose the most visited action
        best_action = np.argmax(visit_counts)
        pi[best_action] = 1.0
    else:
        # Apply temperature
        counts_pow = visit_counts ** (1.0 / temperature)
        sum_counts_pow = np.sum(counts_pow)

        if sum_counts_pow > 1e-6: # Check for non-zero sum to avoid division by zero
            pi = counts_pow / sum_counts_pow
        else:
            # If all visit counts were zero (or very small), fall back to uniform
            # Needs valid actions! For now, return uniform over children that *were* created.
            logger.warning("Sum of powered visit counts is near zero. Falling back.")
            num_children = len(root_node.children)
            if num_children > 0:
                 uniform_prob = 1.0 / num_children
                 for action in root_node.children.keys():
                      if action < num_actions:
                           pi[action] = uniform_prob
            else:
                 # Should not happen if checked earlier, but as safeguard:
                 return pi # Return zeros


    # Final check for NaN or Inf
    if np.isnan(pi).any() or np.isinf(pi).any():
        logger.error(f"Policy contains NaN or Inf! Counts: {visit_counts}, Temp: {temperature}")
        # Fallback to uniform over available children actions
        num_children = len(root_node.children)
        if num_children > 0:
             uniform_prob = 1.0 / num_children
             pi = np.zeros(num_actions, dtype=np.float32)
             for action in root_node.children.keys():
                  if action < num_actions:
                       pi[action] = uniform_prob
        else:
             pi = np.zeros(num_actions, dtype=np.float32) # Zeros if no children

    return pi


def select_action(root_env, model, n_simulations, c_puct, device, temperature=1.0):
    """
    Select an action using MCTS simulation.
    Returns the chosen action and the calculated policy pi.
    """
    num_actions = root_env.configuration.columns
    root_node = make_tree(root_env, model, n_simulations, c_puct, device) # <<< Pass device

    # Create policy pi based on visit counts
    pi = create_pi(root_node, num_actions, temperature)

    # Handle cases where pi might be all zeros (MCTS failed)
    if np.sum(pi) < 1e-6:
        logger.warning("MCTS resulted in a zero policy vector. Choosing a valid action uniformly.")
        valid_actions = get_valid_actions(root_env)
        if not valid_actions:
            logger.error("No valid actions available and MCTS failed. Cannot select action.")
            return None, pi # Indicate failure

        # Create uniform policy over valid actions
        pi = np.zeros(num_actions, dtype=np.float32)
        prob = 1.0 / len(valid_actions)
        for action in valid_actions:
            pi[action] = prob
        # Choose action based on this fallback uniform policy
        action = np.random.choice(valid_actions)
        return action, pi


    # Choose action stochastically based on pi
    try:
        action = np.random.choice(num_actions, p=pi)
    except ValueError as e:
        logger.error(f"Error choosing action with policy pi: {pi}. Error: {e}")
        logger.error(f"Sum of pi: {np.sum(pi)}")
        # Fallback: choose uniformly from actions with non-zero probability in pi
        non_zero_actions = np.where(pi > 1e-6)[0]
        if len(non_zero_actions) > 0:
            action = np.random.choice(non_zero_actions)
            logger.warning(f"Fell back to choosing from non-zero actions: {non_zero_actions}, chose: {action}")
        else:
            # Ultimate fallback: uniform random valid action
            valid_actions = get_valid_actions(root_env)
            if not valid_actions:
                logger.error("No valid actions available and policy sampling failed.")
                return None, pi
            action = np.random.choice(valid_actions)
            logger.warning(f"Fell back to uniform valid action choice: {action}")


    # Ensure chosen action is actually valid (pi should ideally only cover valid actions if priors are masked)
    # MCTS expansion inherently only considers valid moves, so this check might be redundant
    # if create_pi handles invalid actions correctly.
    # valid_actions = get_valid_actions(root_env)
    # if action not in valid_actions:
    #     logger.warning(f"MCTS chose an invalid action {action}. Valid: {valid_actions}. Policy pi: {pi}. Choosing most visited valid action instead.")
    #     # Find the valid action with the highest probability in pi
    #     valid_pi = pi[valid_actions]
    #     if np.sum(valid_pi) > 1e-6:
    #         best_valid_idx = np.argmax(valid_pi)
    #         action = valid_actions[best_valid_idx]
    #     else: # If all valid actions have zero prob (error in pi generation)
    #         action = np.random.choice(valid_actions) # Random valid action


    return action, pi

# --- Example Usage (Optional) ---
if __name__ == "__main__":
    from kaggle_environments import make
    import ConnectXNN as cxnn # Corrected import

    # Setup basic logging if logger_setup is not available
    try:
        from logger_setup import get_logger
    except ImportError:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger("MCTS_Test")

    # <<< GPU Change Start >>>
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info("CUDA device found, using GPU.")
    else:
        dev = torch.device("cpu")
        logger.info("CUDA device not found, using CPU.")
    # <<< GPU Change End >>>


    env = make("connectx", debug=True)
    env.reset()

    # Make a few moves for a non-empty board state
    try:
        env.step([0, 1])
        env.step([2, 3])
    except Exception as e:
        logger.error(f"Error during initial steps: {e}")
        logger.info(f"Current state: {env.state}")


    model = cxnn.ConnectXNet().to(dev) # <<< Move model to device
    model.eval() # Set model to evaluation mode

    logger.info("Running MCTS...")
    selected_action, policy_vector = select_action(
        root_env=env,
        model=model,
        n_simulations=50, # Increase simulations for better test
        c_puct=1.0,
        device=dev, # <<< Pass device
        temperature=1.0
    )

    logger.info(f"Selected Action: {selected_action}")
    logger.info(f"Policy Vector (pi): {policy_vector}")
    logger.info(f"Board state after MCTS:\n{convert_board_to_2D(env)}")

    if selected_action is not None:
        # Example of how to take the step after MCTS decision
        player_idx = 0 if env.state[0]['status'] == 'ACTIVE' else 1
        actions = [None, None]
        actions[player_idx] = selected_action
        try:
            env.step(actions)
            logger.info(f"Board state after taking action {selected_action}:\n{convert_board_to_2D(env)}")
        except Exception as e:
            logger.error(f"Error taking selected action {selected_action}: {e}")
            logger.info(f"State before error: {env.state}")