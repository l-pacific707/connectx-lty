# MCTS_ConnectX.py
"""# MCTS 기반 AlphaZero-style 노드 구조 및 탐색 설계
Based on AlphaZero MCTS node structure and search design.
"""
from logger_setup import get_logger
import math
import numpy as np
from collections import defaultdict
import copy
import torch
import torch.nn.functional as F # Import F for softmax

import ConnectXNN as cxnn


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
        self.W = 0.0    # Total action value (from perspective of player *at this node*)
        self.Q = 0.0    # Mean action value (W / N)
        self.P = prior  # Prior probability from network

    def is_expanded(self):
        return len(self.children) > 0

    def expand(self, action_priors):
        """Expand the node by creating children for valid actions."""
        for action, prob in action_priors:
            if action not in self.children:
                # Child state is initially None; it will be set when the child is selected/simulated
                self.children[action] = MCTSNode(state=None, parent=self, prior=prob)

    def select_child(self, c_puct, log_debug=False):
        """Select the child with the highest UCB score."""
        best_score = -float('inf')
        best_action = None
        best_child = None

        sqrt_total_N = math.sqrt(self.N)

        for action, child in self.children.items():
            # UCB calculation: Q + U
            # Q is the mean action value (W/N) from the perspective of the player *at this node*.
            # U = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            u = c_puct * child.P * sqrt_total_N / (1 + child.N)
            # The child.Q value represents the expected outcome *after* taking 'action',
            # from the perspective of the *child's* player (the opponent).
            # So, we use -child.Q because we want the value from the current node's perspective.
            # Also, child node may have Q = W = N =0 since we just initialize it to be zero until it actually selected to be leaf node.
            if child.N == 0:
                q_term = self.Q*0.9 # FPU applied
            else :
                q_term = child.Q
            score = -q_term + u # Use negative Q of child

            if log_debug:
                 logger.debug(f"  Action {action}: Child_Q={-child.Q:.3f} (raw {-child.Q:.3f}), U={u:.3f}, Score={score:.3f} (N={child.N}, P={child.P:.3f})")


            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        if best_child is None:
            # Use logger.warning which should always be visible
            logger.warning(f"No best child found for node. State:\n{convert_board_to_2D(self.state) if self.state else 'None'}")
            # Handle cases where no valid child might be selectable (e.g., only invalid actions left?)
            # As a fallback, maybe select randomly or the first child?
            if self.children:
                best_action = list(self.children.keys())[0]
                best_child = self.children[best_action]
            else:
                return None, None # Cannot select if no children

        return best_action, best_child

    def backup(self, value, log_debug=False):
        """
        Backup the value estimate through the tree path.
        'value' is the estimated outcome from the perspective of the player
        whose turn it is AT THE LEAF NODE evaluated/simulated.
        """
        node = self
        current_perspective_value = value
        while node is not None:
            # Update the node's statistics
            node.N += 1
            # W tracks the sum of values from the perspective of the player whose turn it is *at this node*.
            # Since 'current_perspective_value' is from the child's perspective,
            # we add it directly here, as W/N = Q should represent the value for the *current* player.
            node.W += current_perspective_value
            node.Q = node.W / node.N

            if log_debug:
                 logger.debug(f"  Backup at node (Player {node.current_player}): N={node.N}, W={node.W:.3f}, Q={node.Q:.3f}, backed_value={current_perspective_value:.3f}")


            # Negate the value for the parent node (opponent's perspective)
            current_perspective_value = -current_perspective_value
            node = node.parent


    def find_current_player(self) -> int:
        """Determine the current player from the environment state."""
        if self.state is None:
             # If state is None, rely on parent (handled in __init__)
             # This function is mainly for the root or error recovery.
             logger.error("find_current_player called on node with no state and no parent info available.")
             return 1 # Default guess or raise error

        # Check Kaggle environment state structure
        try:
            if self.state.state[0]["status"] == "ACTIVE":
                return 1
            elif self.state.state[1]["status"] == "ACTIVE":
                return 2
            else: # Game is DONE or INVALID
                # logger.debug(f"Current state is terminal. State: {self.state.state}") # DEBUG log
                # If terminal, the 'current' player doesn't really matter for selection,
                # but for consistency, determine who *would* have played.
                # Count pieces? More robust but slower.
                board = np.array(self.state.state[0]["observation"]["board"])
                p1_pieces = np.sum(board == 1)
                p2_pieces = np.sum(board == 2)
                # If P1 <= P2 pieces, it's P1's turn (or game just ended on P2's move)
                # If P1 > P2 pieces, it's P2's turn (or game just ended on P1's move)
                return 1 if p1_pieces <= p2_pieces else 2 # Guess based on piece count
        except (AttributeError, IndexError, TypeError) as e:
             logger.error(f"Error accessing state to determine player: {e}. State: {self.state}")
             return 1 # Fallback


def simulate_env(env, action):
    """Creates a deep copy of the environment and performs a step."""
    env_copy = copy.deepcopy(env)
    # Kaggle env step takes a list [player1_action, player2_action]
    # We need to know whose turn it is to place the action correctly.
    try:
        if env_copy.state[0]["status"] == "ACTIVE":
            current_player_idx = 0 # Player 1's turn
        elif env_copy.state[1]["status"] == "ACTIVE":
            current_player_idx = 1 # Player 2's turn
        else:
            # If game is done, stepping might not be valid, but we try anyway
            # Determine who would have moved based on piece count (less reliable)
            board = np.array(env_copy.state[0]["observation"]["board"])
            p1_pieces = np.sum(board == 1)
            p2_pieces = np.sum(board == 2)
            current_player_idx = 0 if p1_pieces <= p2_pieces else 1
            logger.warning(f"Simulating step on a non-ACTIVE game state. Assuming player {current_player_idx+1}'s turn.")

        actions = [None, None]
        actions[current_player_idx] = action
        env_copy.step(actions)
    except Exception as e:
         logger.error(f"Error during simulate_env step: {e}. State before step: {env.state}, Action: {action}", exc_info=True)
         # Return the original env or raise? Returning copy might lead to infinite loops if state doesn't change.
         # Let's return the unmodified copy to signal potential issue.
         return env_copy # Or maybe return None? Returning copy is safer for now.
    return env_copy

def is_terminal(env):
    """Check if the game has ended."""
    return env.done # Kaggle env provides a .done attribute

def get_valid_actions(env):
    """Get a list of valid actions (columns where a piece can be dropped)."""
    try:
        board = np.array(env.state[0]["observation"]["board"])
        columns = env.configuration.columns
        # A column is valid if the top cell (index c) is empty (0)
        return [c for c in range(columns) if board[c] == 0]
    except (AttributeError, IndexError, TypeError) as e:
        logger.error(f"Error getting valid actions: {e}. State: {env.state}")
        return [] # Return empty list on error

def convert_board_to_2D(env):
    """Helper to visualize the board from the 1D list."""
    if not env or not hasattr(env, 'state') or not env.state: return "Invalid env for board conversion"
    try:
        board = np.array(env.state[0]["observation"]["board"])
        rows = env.configuration.rows
        columns = env.configuration.columns
        return str(board.reshape((rows, columns))) # Return as string for logging
    except Exception as e:
        logger.error(f"Error converting board to 2D: {e}")
        return "Error converting board"


def get_game_result(env, perspective_player):
    """
    Determine the game result (+1 win, -1 loss, 0 draw) from the perspective
    of the given player.
    """
    if not env.done:
        logger.error("Requesting game result for a non-terminal state.")
        return 0.0 # Treat as ongoing or draw

    player_index = perspective_player - 1 # 0 for player 1, 1 for player 2

    try:
        # Check status and reward from the perspective_player's state
        status = env.state[player_index]["status"]
        reward = env.state[player_index]["reward"]

        if status == "DONE":
            if reward == 1: # Won
                return 1.0
            elif reward < 0: # Draw or Lost (Kaggle reward is 0 for draw, -1 for loss)
                # Check opponent's reward to distinguish draw/loss
                opponent_index = 1 - player_index
                opponent_reward = env.state[opponent_index]["reward"]
                if opponent_reward == 1: # Opponent won -> Loss for perspective player
                     return -1.0
                else: 
                     logger.warning(f"Unexpected reward value {reward} for player {perspective_player} at game end (Status DONE).")
                     return 0.0
            elif reward == 0.0: # Explicit draw reward
                 return 0.0
            else:
                raise ValueError(f"Unexpected reward value {reward} for player {perspective_player} at game end (Status DONE).")

        elif status == "INVALID":
            logger.warning(f"Game ended with INVALID status for player {perspective_player}.")
            # Penalize the player who caused the invalid move.
            # If perspective_player is the one with INVALID status, they lose.
            # If the *other* player has INVALID status, perspective_player wins.
            opponent_index = 1 - player_index
            opponent_status = env.state[opponent_index]["status"]
            if opponent_status != "INVALID": # Perspective player made the invalid move
                 return -1.0
            else: # Opponent made the invalid move (or both somehow?)
                 # Let's assume opponent's invalid move means perspective player wins
                 # This might need refinement based on how Kaggle handles simultaneous invalid states.
                 return 1.0
        elif status == "ERROR":
             logger.error("Game ended with ERROR status.")
             return 0.0 # Treat as draw or handle differently
        else: # Should not happen (e.g., ACTIVE but env.done is True?)
            logger.error(f"Game is done but status is unexpected: {status}")
            return 0.0
    except (AttributeError, IndexError, TypeError) as e:
         logger.error(f"Error getting game result: {e}. State: {env.state}")
         return 0.0 # Default to draw on error


def make_tree(root_env, model, n_simulations, c_puct, device, log_debug=False):
    """
    Perform MCTS simulations starting from the root_env.

    Args:
        root_env: The starting environment state.
        model: The neural network model.
        n_simulations (int): Number of simulations to run.
        c_puct (float): Exploration constant.
        device: The torch device ('cuda' or 'cpu').
        log_debug (bool): If True, log detailed debug messages.

    Returns:
        MCTSNode: The root node of the search tree after simulations.
    """
    root_node = MCTSNode(state=root_env)
    if log_debug:
        logger.debug(f"Start MCTS. Root state:\n{convert_board_to_2D(root_env)}\nCurrent player: {root_node.current_player}")

    # Initial evaluation and expansion of the root node
    valid_actions = get_valid_actions(root_env)
    if not valid_actions:
         logger.warning("No valid actions from root state. Cannot perform MCTS.")
         return root_node # Return the unexpanded root

    try:
        with torch.no_grad():
            input_tensor = cxnn.preprocess_input(root_env).to(device)
            p_logits, v = model(input_tensor)
            p = torch.softmax(p_logits, dim=-1).detach().cpu().numpy().flatten()
            value_estimate = v.item() # Value from the perspective of the root node's player
            
            #initializing root node statistics to FPU
            root_node.N = 1
            root_node.Q = value_estimate
            root_node.W = value_estimate

        # Filter priors for valid actions ONLY
        action_priors = [(a, p[a]) for a in valid_actions if a < len(p)]
        # Normalize the priors for valid actions? AlphaZero paper doesn't explicitly mention this for expansion.
        # sum_priors = sum(prob for _, prob in action_priors)
        # if sum_priors > 1e-6:
        #     action_priors = [(a, prob / sum_priors) for a, prob in action_priors]

        root_node.expand(action_priors)
        # Backup the initial value estimate? No, backup happens from leaf evaluation.
        if log_debug:
            logger.debug(f"Root node expanded. Initial value estimate: {value_estimate:.4f}")
            logger.debug(f"Root priors (valid): {action_priors}")

    except Exception as e:
         logger.error(f"Error during root node evaluation/expansion: {e}", exc_info=True)
         return root_node # Return unexpanded root on error


    for sim in range(n_simulations):
        if log_debug: logger.debug(f"--- Simulation {sim+1}/{n_simulations} ---")
        node = root_node
        # Use a copy for simulation to avoid modifying the original root_env state object
        simulation_env = copy.deepcopy(root_env)
        search_path = [node] # Keep track of the path for backup

        # --- Selection Phase ---
        while node.is_expanded():
            action, next_node = node.select_child(c_puct, log_debug)
            if action is None or next_node is None:
                 logger.warning(f"Selection failed at simulation {sim+1}. Stopping this sim.")
                 node = None # Mark failure
                 break

            if log_debug: logger.debug(f"Sim {sim+1}: Selected action {action} (Player {node.current_player})")

            # Simulate the action in the environment copy
            simulation_env = simulate_env(simulation_env, action)
            node = next_node # Move to the selected child

            # If the node hasn't been assigned a state yet (it was created during expansion), assign it now.
            if node.state is None:
                node.state = simulation_env # Use the state *after* the action

            search_path.append(node)

            if is_terminal(simulation_env):
                if log_debug: logger.debug(f"Sim {sim+1}: Reached terminal state during selection.")
                break # Move to backup phase

        if node is None: continue # Skip backup if selection failed

        # --- End of Selection (Reached a leaf or terminal state) ---
        leaf_node = node # The last node in the search path
        leaf_value = 0.0 # Value to backup (from perspective of player at leaf_node)

        if is_terminal(simulation_env):
            # Game ended, get the actual outcome
            # The result should be from the perspective of the player whose turn it is *at the leaf node*.
            perspective = leaf_node.current_player
            game_outcome = get_game_result(simulation_env, perspective)
            leaf_value = game_outcome
            if log_debug:
                logger.debug(f"Sim {sim+1}: Terminal state reached. Outcome for P{perspective}: {leaf_value}")
                logger.debug(f"Terminal Board:\n{convert_board_to_2D(simulation_env)}")

        else:
            # --- Expansion & Evaluation Phase (Reached a non-terminal leaf node) ---
            if leaf_node.state is None:
                 logger.error(f"Sim {sim+1}: Reached non-terminal leaf node with no state. Logic error?")
                 # Attempt recovery - use the simulation_env which should be the correct state
                 leaf_node.state = simulation_env
                 # continue # Skip backup for this sim? Better to try and evaluate.

            if log_debug: logger.debug(f"Sim {sim+1}: Reached leaf node. Evaluating state:\n{convert_board_to_2D(leaf_node.state)}")

            try:
                with torch.no_grad():
                    input_tensor = cxnn.preprocess_input(leaf_node.state).to(device)
                    p_logits, v_leaf = model(input_tensor)
                    p_leaf = torch.softmax(p_logits, dim=-1).detach().cpu().numpy().flatten()
                    # v_leaf is the value from the perspective of the player whose turn it is at the leaf node
                    leaf_value = v_leaf.item()

                valid_actions_leaf = get_valid_actions(leaf_node.state)
                if valid_actions_leaf:
                    # Filter priors for valid actions
                    leaf_priors = [(a, p_leaf[a]) for a in valid_actions_leaf if a < len(p_leaf)]
                    # Normalize?
                    leaf_node.expand(leaf_priors)
                    if log_debug:
                         logger.debug(f"Sim {sim+1}: Leaf node expanded with {len(valid_actions_leaf)} actions. Value={leaf_value:.4f}")
                         logger.debug(f"Leaf priors (valid): {leaf_priors}")

                else:
                     # This case means the leaf node has no valid actions, but wasn't detected as terminal earlier.
                     # This implies the game *must* have ended here (either win/loss/draw).
                     if log_debug: logger.debug(f"Sim {sim+1}: Leaf node has no valid actions (Terminal state).")
                     # Recalculate the true outcome as the value.
                     perspective = leaf_node.current_player
                     game_outcome = get_game_result(leaf_node.state, perspective)
                     leaf_value = game_outcome # Override network value with true outcome
                     if log_debug: logger.debug(f"Sim {sim+1}: Overriding leaf value with terminal outcome: {leaf_value}")


            except Exception as e:
                 logger.error(f"Error during leaf node evaluation/expansion: {e}. State:\n{convert_board_to_2D(leaf_node.state)}", exc_info=True)
                 # Cannot evaluate, cannot backup reliably. Skip backup for this sim.
                 continue


        # --- Backup Phase ---
        if log_debug: logger.debug(f"Sim {sim+1}: Backing up value: {leaf_value:.4f} from leaf (Player {leaf_node.current_player})")
        # Backup the value starting from the leaf node
        leaf_node.backup(leaf_value, log_debug)

    return root_node

def create_pi(root_node, num_actions, temperature=1.0):
    """
    Create the policy vector pi based on node visit counts N.
    pi(a|s) = N(s,a)^(1/temp) / sum_b(N(s,b)^(1/temp))
    
    root_node must be fully expanded to make a tree.
    """
    pi = np.zeros(num_actions, dtype=np.float32)
    visit_counts = np.zeros(num_actions, dtype=np.float32)

    if not root_node.children:
        logger.warning("Root node has no children, cannot create policy pi. Returning zeros.")
        # This can happen if MCTS failed (e.g., no valid actions initially).
        return pi # Return zeros, caller must handle based on valid actions.

    # Sum visit counts only for existing children (representing valid actions)
    total_visits = 0
    for action, child in root_node.children.items():
        if 0 <= action < num_actions:
            visit_counts[action] = child.N
            total_visits += child.N
        else:
            logger.warning(f"Action {action} out of bounds (num_actions={num_actions}) in create_pi.")

    # Check if any visits occurred (total_visits should approximately match root_node.N)
    if total_visits == 0 :
         logger.warning("Total visit count of children is zero in create_pi. Returning uniform over children.")
         # Fallback to uniform over the actions that *were* explored (children keys)
         num_children = len(root_node.children)
         if num_children > 0:
              uniform_prob = 1.0 / num_children
              for action in root_node.children.keys():
                   if 0 <= action < num_actions:
                        pi[action] = uniform_prob
         return pi # Return uniform or zeros


    if temperature == 0:
        # Deterministic selection: choose the most visited action among children
        best_action = -1
        max_visits = -1
        for action, child in root_node.children.items():
             if child.N > max_visits:
                  max_visits = child.N
                  best_action = action
        if best_action != -1:
             pi[best_action] = 1.0
    else:
        # Apply temperature to visit counts of children
        powered_counts = {action: child.N ** (1.0 / temperature) for action, child in root_node.children.items()}
        sum_powered_counts = sum(powered_counts.values())

        if sum_powered_counts > 1e-9: # Check for non-zero sum
            for action, p_count in powered_counts.items():
                 if 0 <= action < num_actions:
                      pi[action] = p_count / sum_powered_counts
        else:
            # If all visit counts were zero (or very small), fall back to uniform over children
            logger.warning("Sum of powered visit counts is near zero. Falling back to uniform over children.")
            num_children = len(root_node.children)
            if num_children > 0:
                 uniform_prob = 1.0 / num_children
                 for action in root_node.children.keys():
                      if 0 <= action < num_actions:
                           pi[action] = uniform_prob


    # Final check for NaN or Inf and normalization
    if np.isnan(pi).any() or np.isinf(pi).any():
        logger.error(f"Policy contains NaN or Inf! Counts: {visit_counts}, Temp: {temperature}")
        # Fallback to uniform over available children actions
        num_children = len(root_node.children)
        if num_children > 0:
             uniform_prob = 1.0 / num_children
             pi = np.zeros(num_actions, dtype=np.float32)
             for action in root_node.children.keys():
                  if 0 <= action < num_actions:
                       pi[action] = uniform_prob
        else:
             pi = np.zeros(num_actions, dtype=np.float32) # Zeros if no children
    elif abs(np.sum(pi) - 1.0) > 1e-5:
         # Re-normalize if sum is not close to 1 (can happen with fallbacks)
         current_sum = np.sum(pi)
         if current_sum > 1e-9:
              pi /= current_sum
         else: # If sum is still zero, something went wrong
              logger.error("Policy sum is zero after generation. Cannot normalize.")
              # Fallback to uniform over valid actions if possible
              # This requires access to the environment state, which create_pi doesn't have.
              # Returning the zero vector.
              pi = np.zeros(num_actions, dtype=np.float32)


    return pi


def select_action(root_env, model, n_simulations, c_puct, device, temperature=1.0, log_debug=False):
    """
    Select an action using MCTS simulation.

    Args:
        root_env: The current environment state.
        model: The neural network model.
        n_simulations (int): Number of MCTS simulations.
        c_puct (float): Exploration constant.
        device: Torch device.
        temperature (float): Temperature for sampling action from policy.
        log_debug (bool): Whether to log MCTS debug messages.

    Returns:
        tuple: (chosen_action, policy_vector_pi) or (None, zero_policy) on failure.
    """
    num_actions = root_env.configuration.columns
    # Run MCTS
    root_node = make_tree(root_env, model, n_simulations, c_puct, device, log_debug)

    # Create policy pi based on visit counts
    pi = create_pi(root_node, num_actions, temperature)

    if log_debug:
        logger.debug(f"MCTS complete. Root N={root_node.N}, Q={root_node.Q:.4f}")
        # Log child visit counts and final policy
        child_info = {a: (c.N, c.Q, pi[a]) for a, c in root_node.children.items() if 0 <= a < num_actions}
        logger.debug(f"Child N/Q/Pi: {child_info}")
        logger.debug(f"Final policy pi (sum={np.sum(pi):.4f}): {np.round(pi, 3)}")


    # Handle cases where pi might be all zeros (MCTS failed or no valid moves)
    valid_actions = get_valid_actions(root_env)
    if not valid_actions:
         logger.error("No valid actions available. Cannot select action.")
         return None, np.zeros(num_actions, dtype=np.float32) # Indicate failure

    if np.sum(pi) < 1e-6:
        logger.warning("MCTS resulted in a zero policy vector. Choosing a valid action uniformly.")
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
        # Ensure probabilities sum to 1 for np.random.choice
        pi_normalized = pi / np.sum(pi)
        action = np.random.choice(num_actions, p=pi_normalized)
    except ValueError as e:
        logger.error(f"Error choosing action with policy pi: {pi}. Sum: {np.sum(pi)}. Error: {e}")
        # Fallback: choose uniformly from actions with non-zero probability in pi
        non_zero_actions = np.where(pi > 1e-9)[0]
        if len(non_zero_actions) > 0:
            action = np.random.choice(non_zero_actions)
            logger.warning(f"Fell back to choosing from non-zero actions: {non_zero_actions}, chose: {action}")
        else:
            # Ultimate fallback: uniform random valid action
             action = np.random.choice(valid_actions)
             logger.warning(f"Fell back to uniform valid action choice: {action}")


    # Final check: Ensure chosen action is valid
    if action not in valid_actions:
        logger.warning(f"MCTS chose an invalid action {action}. Valid: {valid_actions}. Policy pi: {pi}. Choosing most visited valid action instead.")
        # Find the valid action with the highest probability in pi
        valid_pi = {a: pi[a] for a in valid_actions if 0 <= a < len(pi)}
        if valid_pi:
             action = max(valid_pi, key=valid_pi.get)
        else: # If all valid actions have zero prob (error in pi generation)
            action = np.random.choice(valid_actions) # Random valid action


    return action, pi

# --- Example Usage (Optional) ---
if __name__ == "__main__":
    from kaggle_environments import make
    # import ConnectXNN as cxnn # Already imported

    # Setup basic logging if logger_setup is not available
    try:
        from logger_setup import get_logger
    except ImportError:
        import logging
        logging.basicConfig(level=logging.DEBUG) # Set level to DEBUG to see MCTS logs
        logger = logging.getLogger("MCTS_Test")

    # Determine device
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info("CUDA device found, using GPU.")
    else:
        dev = torch.device("cpu")
        logger.info("CUDA device not found, using CPU.")


    env = make("connectx", debug=True) # Use debug=True for env checks
    env.reset()

    # Make a few moves for a non-empty board state
    try:
        env.step([0, None]) 
        env.step([None, 4]) 
        env.step([0, None])
        env.step([None, 4])
        env.step([2, None])
        env.step([None, 4])
        logger.info(f"Initial board state for MCTS test:\n{convert_board_to_2D(env)}")
    except Exception as e:
        logger.error(f"Error during initial steps: {e}")
        logger.info(f"Current state: {env.state}")

    model1 = cxnn.ConnectXNet(num_res_blocks=5)
    model1 = cxnn.load_model(model1, path = "./models/best", filename = "best_model.pth") # Load your model
    model1.eval() # Set model to evaluation mode
    
    model2 = cxnn.ConnectXNet(num_res_blocks=5)
    model2 = cxnn.load_model(model2, path = "./models/checkpoints", filename = "model_iter_60.pth") # Load your model
    model2.eval() # Set model to evaluation mode
    
    while not env.done:
        player_mark = env.state[0]['observation']['mark'] # 1 or 2
        player_idx = player_mark - 1 # 0 or 1
        if player_mark == 1:
            model = model1
        elif player_mark == 2:
            model = model2
        else:
            raise ValueError(f"Invalid player mark: {player_mark}")
        selected_action, policy_vector = select_action(
            root_env=env, 
            model=model,
            n_simulations=120, # Increase simulations for better test
            c_puct=1.5,
            device=dev,
            temperature=0.8,
            log_debug=True # Enable debug logging for this test run
        )

        logger.info(f"Selected Action: {selected_action}")
        logger.info(f"Policy Vector (pi): {policy_vector}")
        logger.info(f"Board state before taking action:\n{convert_board_to_2D(env)}")

        if selected_action is not None:
            # Example of how to take the step after MCTS decision
            # Need to know whose turn it is
            actions = [None, None]
            actions[player_idx] = selected_action
            try:
                env.step(actions)
                logger.info(f"Board state after taking action {selected_action}:\n{convert_board_to_2D(env)}")
            except Exception as e:
                logger.error(f"Error taking selected action {selected_action}: {e}")
                logger.info(f"State before error: {env.state}")
