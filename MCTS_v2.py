# MCTS_Connectx_modified.py
import math
import numpy as np
import copy
import torch
import torch.nn.functional as F

# Assuming ConnectXNN is in the same directory or accessible in PYTHONPATH
# For MCTS internal use, we'll define a helper to create NN input
# import ConnectXNN as cxnn # This would be used by the training script, not directly in MCTS logic if env is removed

from logger_setup import get_logger # Assuming logger_setup.py is available

logger = get_logger("MCTS_Mod", "MCTS_Mod.log")

# --- Game Logic Helper Functions ---

def _check_line(board, player, r_start, c_start, dr, dc, config):
    """Checks if a player has a winning line from (r_start, c_start) in direction (dr, dc)."""
    rows, cols, inarow = config['rows'], config['columns'], config['inarow']
    count = 0
    for i in range(inarow):
        r, c = r_start + i * dr, c_start + i * dc
        if 0 <= r < rows and 0 <= c < cols and board[r, c] == player:
            count += 1
        else:
            break
    return count == inarow

def _check_win(board, player, config):
    """Checks if the given player has won."""
    rows, cols = config['rows'], config['columns']
    # Directions: horizontal, vertical, diagonal_down_right, diagonal_up_right
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(rows):
        for c in range(cols):
            if board[r, c] == player:
                for dr, dc in directions:
                    # Only need to check starting points that could form a line
                    # For horizontal, only check if c <= cols - inarow
                    # For vertical, only check if r <= rows - inarow
                    # Similar logic for diagonals
                    if _check_line(board, player, r, c, dr, dc, config):
                        return True
    return False

def _is_board_full(board, config):
    """Checks if the board is full (no empty cells)."""
    return not np.any(board == 0)

def get_terminal_state_and_winner(board, config):
    """
    Checks if the game has ended and returns the winner.
    Returns:
        tuple: (is_terminal (bool), winner (int or None))
               winner = 1 if player 1 wins
               winner = 2 if player 2 wins
               winner = 0 for a draw
               winner = -1 if game is not terminal (or None for consistency, let's use -1)
    """
    if _check_win(board, 1, config):
        return True, 1
    if _check_win(board, 2, config):
        return True, 2
    if _is_board_full(board, config):
        return True, 0  # Draw
    return False, -1  # Game not terminal

def _get_valid_actions(board, config):
    """Gets a list of valid actions (columns where a piece can be dropped)."""
    return [c for c in range(config['columns']) if board[0, c] == 0]

def _apply_action(board, column_action, player, config):
    """
    Applies an action to the board and returns the new board state.
    Returns None if the action is invalid.
    """
    if column_action not in _get_valid_actions(board, config):
        logger.error(f"Invalid action {column_action} attempted for player {player}.")
        return None

    new_board = board.copy()
    for r in range(config['rows'] - 1, -1, -1):
        if new_board[r, column_action] == 0:
            new_board[r, column_action] = player
            return new_board
    return None # Should not happen if action is valid

def _convert_board_to_2D_string(board, config):
    """Helper to visualize the board."""
    if board is None: return "None board"
    return str(board)

def _create_nn_input(board, current_player_to_act, config):
    """
    Prepares the board state for neural network input.
    Output: torch.Tensor of shape (1, 3, board_height, board_width).
    Channels: (Player 1 stones, Player 2 stones, Player-to-move indicator)
    """
    p1_plane = (board == 1).astype(np.float32)
    p2_plane = (board == 2).astype(np.float32)

    # Player-to-move plane: 1 if current_player_to_act is P1, -1 if current_player_to_act is P2
    turn_plane_val = 1.0 if current_player_to_act == 1 else -1.0
    turn_plane = np.full((config['rows'], config['columns']), turn_plane_val, dtype=np.float32)

    stacked = np.stack([p1_plane, p2_plane, turn_plane])
    return torch.from_numpy(stacked).unsqueeze(0)


class MCTSNode:
    def __init__(self, board, player_to_act, config, parent=None, prior_prob=0.0):
        self.board = board  # NumPy array representing the board
        self.player_to_act = player_to_act  # Player whose turn it is at this node (1 or 2)
        self.config = config # Game configuration
        self.parent = parent
        self.children = {}  # action: MCTSNode

        self.N = 0  # Visit count
        self.W = 0.0  # Total action value (from perspective of self.player_to_act)
        self.V = 0.0  # Mean action value (W / N)
        self.P = prior_prob  # Prior probability from network

    def is_expanded(self):
        return len(self.children) > 0

    def expand(self, action_priors):
        """Expand the node by creating children for valid actions."""
        # action_priors: list of (action, prob) tuples
        next_player = 3 - self.player_to_act # Switch player (1->2, 2->1)
        for action, prob in action_priors:
            if action not in self.children:
                # Board state of child will be set when selected, or pass None for now
                self.children[action] = MCTSNode(board=None, # Child board set on first visit
                                                 player_to_act=next_player,
                                                 config=self.config,
                                                 parent=self,
                                                 prior_prob=prob)

    def select_child(self, c_puct, c_fpu, log_debug=False):
        """Select the child with the highest UCB1 score."""
        best_score = -float('inf')
        best_action = None
        best_child = None

        sqrt_total_N_parent = math.sqrt(self.N)
        
        # Calculate sum of priors of already explored children for FPU
        # explored_childs_prior_sum = sum(child.P for child in self.children.values() if child.N > 0)

        for action, child in self.children.items():
            if child.N == 0: # First Play Urgency (FPU) for unvisited children
                # AlphaZero uses Q_value of parent + U_value for unvisited children
                # A common FPU modification makes Q slightly worse for unvisited nodes
                # Or use parent's value as an estimate (child.V is 0)
                # Original AlphaZero paper used Q=0 for unvisited children in the PUCT formula.
                # Let's stick to a simpler PUCT variant or a common FPU strategy.
                # For now, if N=0, Q_child_perspective effectively is 0 unless parent's V is used.
                # The term `self.V` below is from the parent's perspective.
                # If we want Q from child's perspective, and child is unvisited, it's often estimated as 0
                # or parent_value_from_child_perspective (-self.V)
                # The provided code had `q_term = self.V - c_fpu * explored_childs_prior`
                # self.V is Q-value for current player at THIS node.
                # A simple FPU might be: q_child = -self.V (parent's value from child's view) + some_bonus_or_malus
                # Let's use a common PUCT variant where unvisited Q is 0 or a fixed value.
                # For simplicity, if child.N = 0, child.V = 0, so -child.V = 0.
                q_term_child_perspective = 0.0 # Default Q for unvisited node from child's perspective
                if c_fpu > 0: # Apply FPU only if c_fpu is positive
                    # A simple FPU: give a slight malus or use parent's value adjusted
                    # Using parent's value from current player's perspective (self.V)
                    # and adjusting it with FPU term.
                    # explored_childs_prior_sum is tricky, let's use simpler FPU for now:
                    q_term_child_perspective = -self.V + c_fpu * child.P * sqrt_total_N_parent / (1 + child.N) # child.N is 0 here
                    # This is essentially giving priority to exploration (U term)
                    # and parent's value. A more standard FPU approach often involves a fixed value.
                    # The original reference code's FPU (`self.V - c_fpu * explored_prior`)
                    # was from parent's perspective. If we evaluate from child's perspective for Q term:
                    # Q_child = -Value_parent. Let's assume Q for unvisited child is 0 for now to simplify.
                    pass # Current Q term from child perspective will be 0 if child.N=0


            else:
                q_term_child_perspective = child.W / child.N # This is child.V

            # UCB calculation: Q(s',a') + U(s',a')
            # Q value is from the perspective of the player whose turn it is AT THE CHILD node.
            # So, child.V (or child.W / child.N) is already in the child's perspective.
            # PUCT formula: V(s_child) + c_puct * P(s_parent, a_to_child) * sqrt(N(s_parent)) / (1 + N(s_child))
            # Here, self is s_parent. child is s_child.
            # V(s_child) needs to be from parent's perspective, so use -child.V.
            
            q_value_for_parent = -q_term_child_perspective # Value of child from parent's perspective
            
            u_value = c_puct * child.P * sqrt_total_N_parent / (1 + child.N)
            score = q_value_for_parent + u_value

            if log_debug:
                logger.debug(f"  Action {action}: Child_N={child.N}, Child_W={child.W:.3f}, Child_V_own_persp={q_term_child_perspective:.3f}, "
                             f"Q_parent_persp={q_value_for_parent:.3f}, U={u_value:.3f}, Score={score:.3f} (P={child.P:.3f})")

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        if best_child is None:
            logger.warning(f"No best child found for node. Board:\n{_convert_board_to_2D_string(self.board, self.config)}")
            if self.children: # Fallback if all scores were -inf (should not happen with proper P)
                best_action = list(self.children.keys())[0]
                best_child = self.children[best_action]
                logger.warning(f"Fallback: selected first child action {best_action}")
            else:
                return None, None

        return best_action, best_child


    def backup(self, value_estimate_from_leaf_player_perspective, log_debug=False):
        """
        Backup the value estimate through the tree path.
        'value_estimate_from_leaf_player_perspective' is the estimated outcome
        from the perspective of the player whose turn it is AT THE LEAF NODE evaluated.
        """
        node = self
        current_perspective_value = value_estimate_from_leaf_player_perspective

        while node is not None:
            node.N += 1
            # W tracks the sum of values from the perspective of node.player_to_act.
            # If current_perspective_value is from child's perspective (node.player_to_act's opponent),
            # then we need to add -current_perspective_value.
            # However, the 'value' passed to backup() should be aligned with node.player_to_act
            # at each step of the recursion.
            # The value passed to parent should be negated.
            node.W += current_perspective_value
            node.V = node.W / node.N # V is always W/N from node.player_to_act's view

            if log_debug:
                logger.debug(f"  Backup at node (Player {node.player_to_act}): "
                             f"N={node.N}, W={node.W:.3f}, V={node.V:.3f}, "
                             f"backed_value_for_this_node={current_perspective_value:.3f}")

            current_perspective_value = -current_perspective_value # Negate for parent
            node = node.parent


def get_game_result_for_player(board, config, perspective_player):
    """
    Determine the game result (+1 win, -1 loss, 0 draw) from the perspective
    of the given player.
    """
    is_done, winner = get_terminal_state_and_winner(board, config)

    if not is_done:
        # logger.error("Requesting game result for a non-terminal state.") # Should not happen if called correctly
        return 0.0 # Treat as ongoing

    if winner == perspective_player:
        return 1.0
    elif winner == 0: # Draw
        return 0.0
    elif winner == -1: # Should be caught by "not is_done"
        logger.error("Terminal state flagged but no winner/draw identified, or winner is -1 erroneously.")
        return 0.0
    else: # Opponent won
        return -1.0


def make_tree(initial_board, initial_player_to_act, config,
              model, n_simulations, c_puct, c_fpu, device,
              np_rng, dirichlet_alpha, dirichlet_epsilon, log_debug=False):
    """
    Perform MCTS simulations starting from the initial_board and initial_player_to_act.
    """
    root_node = MCTSNode(board=initial_board, player_to_act=initial_player_to_act, config=config)

    if log_debug:
        logger.debug(f"Start MCTS. Root board:\n{_convert_board_to_2D_string(initial_board, config)}\n"
                     f"Current player: {initial_player_to_act}")

    # Initial evaluation and expansion of the root node
    valid_actions_root = _get_valid_actions(initial_board, config)
    if not valid_actions_root:
        logger.warning("No valid actions from root state. Cannot perform MCTS.")
        return root_node # Return the unexpanded root

    try:
        with torch.no_grad():
            nn_input_tensor = _create_nn_input(initial_board, initial_player_to_act, config).to(device)
            p_logits, v_nn = model(nn_input_tensor) # p_logits: (1, num_actions), v_nn: (1,1)
            policy_priors_all = torch.softmax(p_logits, dim=-1).squeeze(0).cpu().numpy() # (num_actions,)
            value_estimate_for_root_player = v_nn.item() # From perspective of initial_player_to_act

        # Filter priors for valid actions and apply Dirichlet noise
        action_priors_valid = []
        raw_priors_for_noise = []
        for action in valid_actions_root:
            if 0 <= action < len(policy_priors_all):
                raw_priors_for_noise.append(policy_priors_all[action])
                action_priors_valid.append((action, policy_priors_all[action])) # Store (action, raw_prior)
            else:
                 logger.warning(f"Action {action} from valid_actions out of bounds for policy_priors (len {len(policy_priors_all)})")


        if dirichlet_alpha > 0 and len(raw_priors_for_noise) > 0:
            noise = np_rng.dirichlet([dirichlet_alpha] * len(raw_priors_for_noise))
            
            final_action_priors = []
            for i, (action, raw_prior) in enumerate(action_priors_valid):
                noisy_prior = (1 - dirichlet_epsilon) * raw_prior + dirichlet_epsilon * noise[i]
                final_action_priors.append((action, noisy_prior))
        else:
            final_action_priors = action_priors_valid


        # Initialize root node stats (AlphaZero does this after first eval)
        # N=1 because we've "visited" it once for evaluation
        root_node.N = 1 
        root_node.W = value_estimate_for_root_player 
        root_node.V = value_estimate_for_root_player

        root_node.expand(final_action_priors)

        if log_debug:
            logger.debug(f"Root node expanded. Initial value (for P{initial_player_to_act}): {value_estimate_for_root_player:.4f}")
            # logger.debug(f"Root priors (raw valid): {action_priors_valid}")
            logger.debug(f"Root priors (after noise, for expansion): {final_action_priors}")


    except Exception as e:
        logger.error(f"Error during root node evaluation/expansion: {e}", exc_info=True)
        return root_node

    for sim_idx in range(n_simulations):
        if log_debug: logger.debug(f"--- Simulation {sim_idx + 1}/{n_simulations} ---")
        
        current_node = root_node
        # Board state for current simulation path. Start with a copy of the root board.
        sim_board = initial_board.copy()
        # search_path = [current_node] # Not strictly needed with parent pointers for backup

        # --- Selection Phase ---
        while current_node.is_expanded():
            action, next_node_template = current_node.select_child(c_puct, c_fpu, log_debug)

            if action is None or next_node_template is None:
                logger.warning(f"Selection failed at simulation {sim_idx + 1}. Node has children but select_child returned None. Stopping this sim.")
                current_node = None # Mark failure
                break
            
            if log_debug: logger.debug(f"Sim {sim_idx + 1}: Selected action {action} (Player {current_node.player_to_act} at current_node)")

            # Apply action to get new board for the child
            new_sim_board = _apply_action(sim_board, action, current_node.player_to_act, config)
            if new_sim_board is None:
                logger.error(f"Sim {sim_idx + 1}: Applying action {action} returned None board. This means action was invalid despite being selected.")
                current_node = None # Mark failure
                break
            sim_board = new_sim_board
            
            current_node = next_node_template # Move to the selected child template
            
            # If this child node is being "visited" for the first time in this path,
            # its board state needs to be set.
            if current_node.board is None:
                current_node.board = sim_board # Assign the actual board state to the child node
            # search_path.append(current_node)

            is_terminal_leaf, _ = get_terminal_state_and_winner(sim_board, config)
            if is_terminal_leaf:
                if log_debug: logger.debug(f"Sim {sim_idx + 1}: Reached terminal state during selection.")
                break
        
        if current_node is None: continue # Skip backup if selection or action application failed

        # --- End of Selection (Reached a leaf or terminal state) ---
        leaf_node = current_node
        value_for_leaf_player = 0.0

        is_terminal_final, winner_final = get_terminal_state_and_winner(leaf_node.board, config)

        if is_terminal_final:
            # Game ended, get the actual outcome from perspective of player_to_act at leaf_node
            value_for_leaf_player = get_game_result_for_player(leaf_node.board, config, leaf_node.player_to_act)
            if log_debug:
                logger.debug(f"Sim {sim_idx + 1}: Terminal leaf reached. Winner: {winner_final}. "
                             f"Value for P{leaf_node.player_to_act}: {value_for_leaf_player:.4f}")
                logger.debug(f"Terminal Board:\n{_convert_board_to_2D_string(leaf_node.board, config)}")
        else:
            # --- Expansion & Evaluation Phase (Reached a non-terminal unexpanded leaf node) ---
            if log_debug: logger.debug(f"Sim {sim_idx + 1}: Reached non-terminal leaf. Evaluating board:\n"
                                     f"{_convert_board_to_2D_string(leaf_node.board, config)}")
            try:
                with torch.no_grad():
                    nn_input_tensor = _create_nn_input(leaf_node.board, leaf_node.player_to_act, config).to(device)
                    p_logits_leaf, v_nn_leaf = model(nn_input_tensor)
                    policy_priors_leaf_all = torch.softmax(p_logits_leaf, dim=-1).squeeze(0).cpu().numpy()
                    value_for_leaf_player = v_nn_leaf.item() # From perspective of leaf_node.player_to_act

                valid_actions_leaf = _get_valid_actions(leaf_node.board, config)
                if valid_actions_leaf:
                    leaf_action_priors = []
                    for act_leaf in valid_actions_leaf:
                         if 0 <= act_leaf < len(policy_priors_leaf_all):
                            leaf_action_priors.append((act_leaf, policy_priors_leaf_all[act_leaf]))
                    
                    # No Dirichlet noise for non-root expansions in AlphaZero generally
                    leaf_node.expand(leaf_action_priors)
                    if log_debug:
                        logger.debug(f"Sim {sim_idx + 1}: Leaf node (P{leaf_node.player_to_act}) expanded with {len(valid_actions_leaf)} actions. "
                                     f"NN Value={value_for_leaf_player:.4f}")
                        # logger.debug(f"Leaf priors (valid): {leaf_action_priors}")
                else:
                    # This case means the leaf has no valid moves but wasn't caught as terminal.
                    # This implies it IS terminal (e.g., board full but check_win missed it, or logic error).
                    # Re-evaluate as terminal.
                    logger.warning(f"Sim {sim_idx + 1}: Leaf node has no valid actions but not initially terminal. Re-evaluating.")
                    is_term_recheck, winner_recheck = get_terminal_state_and_winner(leaf_node.board, config)
                    if is_term_recheck:
                        value_for_leaf_player = get_game_result_for_player(leaf_node.board, config, leaf_node.player_to_act)
                        if log_debug: logger.debug(f"Sim {sim_idx + 1}: Re-evaluated as terminal. Winner: {winner_recheck}. New value: {value_for_leaf_player:.4f}")
                    else:
                        logger.error(f"Sim {sim_idx + 1}: Leaf has no valid actions but STILL not terminal after recheck. Logic error in game rules or MCTS.")
                        value_for_leaf_player = 0.0 # Default to draw on error


            except Exception as e:
                logger.error(f"Error during leaf node NN evaluation/expansion: {e}. Board:\n"
                             f"{_convert_board_to_2D_string(leaf_node.board, config)}", exc_info=True)
                continue # Skip backup for this simulation

        # --- Backup Phase ---
        if log_debug: logger.debug(f"Sim {sim_idx + 1}: Backing up value: {value_for_leaf_player:.4f} "
                                 f"from leaf (Player {leaf_node.player_to_act} perspective)")
        leaf_node.backup(value_for_leaf_player, log_debug)

    return root_node


def create_mcts_policy(root_node, temperature=1.0):
    """
    Create the policy vector pi based on node visit counts N.
    pi(a|s) = N(s,a)^(1/temp) / sum_b(N(s,b)^(1/temp))
    """
    num_actions = root_node.config['columns']
    pi = np.zeros(num_actions, dtype=np.float32)
    
    if not root_node.children:
        logger.warning("Root node has no children, cannot create policy pi. This might mean no valid moves or MCTS error.")
        # If there are no children because no valid moves, this is okay.
        # The caller (select_action) should handle this by checking valid_actions again.
        return pi

    child_visit_counts = np.array([root_node.children.get(action, MCTSNode(None,None,None)).N # Access N, default to 0 if action not in children
                                  for action in range(num_actions)], dtype=np.float32)

    if np.sum(child_visit_counts) == 0: # Should not happen if root_node.N > 0 and expanded
        logger.warning("Sum of child visit counts is zero in create_mcts_policy. Falling back to uniform over existing children.")
        # This implies no simulations explored any children, or root_node wasn't properly processed.
        # Fallback to uniform over the actions that *were* explored (children keys)
        num_children_actually_present = len(root_node.children)
        if num_children_actually_present > 0:
            uniform_prob = 1.0 / num_children_actually_present
            for action_key in root_node.children.keys():
                if 0 <= action_key < num_actions:
                    pi[action_key] = uniform_prob
        return pi


    if temperature == 0: # Deterministic: choose the most visited action
        # Handle ties by choosing the first one among the max
        # Filter to only existing children to get the correct most visited action
        max_visits = -1
        best_action = -1
        for action_idx, child_node in root_node.children.items():
            if child_node.N > max_visits:
                max_visits = child_node.N
                best_action = action_idx
        
        if best_action != -1: # If there was at least one child
            pi[best_action] = 1.0
        else: # No children were actually visited (e.g. root_node.children empty or all N=0)
            logger.error("Temperature is 0 but no best action found (all child N=0 or no children). Returning zero policy.")

    else:
        powered_counts = child_visit_counts ** (1.0 / temperature)
        sum_powered_counts = np.sum(powered_counts)
        if sum_powered_counts > 1e-9:
            pi = powered_counts / sum_powered_counts
        else: # sum is zero, likely all child_visit_counts are zero
            logger.warning("Sum of powered visit counts is near zero. This means children were not visited or temperature is problematic.")
            # Fallback: if root_node.children exists, distribute uniformly among them
            num_actual_children = len(root_node.children)
            if num_actual_children > 0:
                uniform_prob = 1.0 / num_actual_children
                for action_key in root_node.children.keys():
                     if 0 <= action_key < num_actions:
                          pi[action_key] = uniform_prob
            # else pi remains zeros

    # Final check for NaN or Inf and normalization (should be less likely with np operations)
    if np.isnan(pi).any() or np.isinf(pi).any():
        logger.error(f"Policy contains NaN or Inf! Counts: {child_visit_counts}, Temp: {temperature}. Resetting to uniform.")
        # Fallback to uniform over available children actions
        num_actual_children = len(root_node.children)
        if num_actual_children > 0:
            uniform_prob = 1.0 / num_actual_children
            pi = np.zeros(num_actions, dtype=np.float32)
            for action_key in root_node.children.keys():
                if 0 <= action_key < num_actions:
                    pi[action_key] = uniform_prob
        else:
            pi = np.zeros(num_actions, dtype=np.float32) # Zeros if no children
    elif abs(np.sum(pi) - 1.0) > 1e-5 and np.sum(pi) > 1e-9 : # If sum is not 1 but also not 0
         logger.warning(f"Policy sum is {np.sum(pi)}, not 1.0. Re-normalizing.")
         pi /= np.sum(pi)
    elif np.sum(pi) < 1e-9 and len(root_node.children) > 0 : # Sum is 0 but there are children means something is wrong.
         logger.error(f"Policy sum is zero despite children existing. Check logic. Resetting to uniform over children.")
         num_actual_children = len(root_node.children)
         uniform_prob = 1.0 / num_actual_children
         pi = np.zeros(num_actions, dtype=np.float32)
         for action_key in root_node.children.keys():
            if 0 <= action_key < num_actions:
                pi[action_key] = uniform_prob


    return pi


def select_mcts_action(initial_board, initial_player_to_act, config,
                       model, n_simulations, c_puct, c_fpu, device,
                       np_rng, dirichlet_alpha, dirichlet_epsilon,
                       temperature=1.0, log_debug=False):
    """
    Select an action using MCTS simulation.
    Returns:
        tuple: (chosen_action (int), policy_vector_pi (np.array))
               Returns (None, zero_policy) on failure or if no valid actions.
    """
    num_actions = config['columns']
    
    # Check for valid actions before starting MCTS.
    # If no valid actions, game is over or something is wrong.
    current_valid_actions = _get_valid_actions(initial_board, config)
    if not current_valid_actions:
        logger.error(f"No valid actions available for P{initial_player_to_act} on board:\n"
                     f"{_convert_board_to_2D_string(initial_board, config)}. Cannot select action.")
        return None, np.zeros(num_actions, dtype=np.float32)


    root = make_tree(initial_board, initial_player_to_act, config,
                     model, n_simulations, c_puct, c_fpu, device,
                     np_rng, dirichlet_alpha, dirichlet_epsilon, log_debug)

    pi_policy = create_mcts_policy(root, temperature)

    if log_debug:
        logger.debug(f"MCTS complete. Root (P{root.player_to_act}) N={root.N}, V={root.V:.4f}")
        child_info_log = {}
        for action_idx, child_node_obj in sorted(root.children.items()):
             child_info_log[action_idx] = (f"N:{child_node_obj.N}", f"V_own:{child_node_obj.V:.3f}", f"P_prior:{child_node_obj.P:.3f}", f"Pi_final:{pi_policy[action_idx]:.3f}")
        logger.debug(f"Children Info (Action: (N, V_own_persp, P_prior, Pi_final)): {child_info_log}")
        logger.debug(f"Final policy pi (sum={np.sum(pi_policy):.4f}, temp={temperature}): {np.round(pi_policy, 3)}")


    # If pi_policy sum is near zero (e.g., MCTS failed or no valid moves explored),
    # fall back to uniform random choice among valid actions.
    if np.sum(pi_policy) < 1e-6:
        logger.warning("MCTS resulted in a near-zero policy vector. Choosing a valid action uniformly.")
        if not current_valid_actions: # Should have been caught earlier
            return None, np.zeros(num_actions, dtype=np.float32)
            
        pi_policy = np.zeros(num_actions, dtype=np.float32)
        prob_uniform = 1.0 / len(current_valid_actions)
        for action_val in current_valid_actions:
            pi_policy[action_val] = prob_uniform
        
        chosen_action = np_rng.choice(current_valid_actions) # Choose based on this new uniform policy
        return chosen_action, pi_policy

    # Choose action stochastically based on pi_policy
    try:
        # Ensure probabilities sum to 1 for np.random.choice, can be slightly off due to float precision
        pi_normalized = pi_policy / np.sum(pi_policy)
        chosen_action = np_rng.choice(num_actions, p=pi_normalized)
    except ValueError as e: # If pi_policy contains NaN or sums to zero leading to error
        logger.error(f"Error choosing action with policy pi: {pi_policy}. Sum: {np.sum(pi_policy)}. Error: {e}")
        logger.warning("Falling back to uniform choice among valid actions due to policy error.")
        if not current_valid_actions: return None, np.zeros(num_actions, dtype=np.float32)
        chosen_action = np_rng.choice(current_valid_actions)
        # Reconstruct pi_policy as uniform for return consistency
        pi_policy = np.zeros(num_actions, dtype=np.float32)
        prob_uniform = 1.0 / len(current_valid_actions)
        for action_val in current_valid_actions:
            pi_policy[action_val] = prob_uniform
        return chosen_action, pi_policy


    # Final check: Ensure chosen action is valid (should be, if pi_policy was based on valid children)
    if chosen_action not in current_valid_actions:
        logger.warning(f"MCTS chose an invalid action {chosen_action}. Valid: {current_valid_actions}. Policy pi: {pi_policy}.")
        # Fallback: choose the valid action with the highest probability in pi_policy
        highest_prob_valid_action = -1
        max_prob_for_valid = -1.0
        for valid_act in current_valid_actions:
            if pi_policy[valid_act] > max_prob_for_valid:
                max_prob_for_valid = pi_policy[valid_act]
                highest_prob_valid_action = valid_act
        
        if highest_prob_valid_action != -1:
            chosen_action = highest_prob_valid_action
            logger.warning(f"Fell back to choosing most probable valid action: {chosen_action}")
        else: # All valid actions had zero probability (error in pi generation or no valid actions)
            logger.warning("All valid actions have zero probability in policy. Choosing random valid action.")
            if not current_valid_actions: return None, np.zeros(num_actions, dtype=np.float32) # Should be caught
            chosen_action = np_rng.choice(current_valid_actions)

    return chosen_action, pi_policy


# --- Example Usage (Test stub) ---
if __name__ == "__main__":
    # Assuming ConnectXNN.py is in the same directory or python path
    try:
        from ConnectXNN import ConnectXNet, load_model
    except ImportError:
        logger.error("ConnectXNN.py not found. Please ensure it's in the correct path.")
        # Define a dummy model for testing MCTS logic if ConnectXNN is not available
        class DummyConnectXNet(torch.nn.Module):
            def __init__(self, n_actions=7):
                super().__init__()
                self.n_actions = n_actions
                self.fc_p = torch.nn.Linear(1, n_actions) # Dummy
                self.fc_v = torch.nn.Linear(1, 1)         # Dummy
            def forward(self, x): # x shape (B, C, H, W)
                # Dummy output: uniform policy, value 0
                batch_size = x.shape[0]
                policy_logits = torch.ones(batch_size, self.n_actions) / self.n_actions
                value = torch.zeros(batch_size, 1)
                return policy_logits, value
        ConnectXNet = DummyConnectXNet
        def load_model(model, path, filename, device=None): return model # Dummy load


    logger.info("--- MCTS (Modified) Test Run ---")
    
    game_config = {'rows': 6, 'columns': 7, 'inarow': 4}
    test_board = np.zeros((game_config['rows'], game_config['columns']), dtype=int)
    current_player = 1

    # Make a few moves for a non-empty board state
    test_board = _apply_action(test_board, 3, 1, game_config) # P1
    if test_board is not None: test_board = _apply_action(test_board, 3, 2, game_config) # P2
    if test_board is not None: test_board = _apply_action(test_board, 4, 1, game_config) # P1
    if test_board is not None: test_board = _apply_action(test_board, 4, 2, game_config) # P2
    if test_board is not None: test_board = _apply_action(test_board, 5, 1, game_config) # P1
    # P2's turn, P1 has 3-in-a-row horizontally (3,4,5)
    # P2 must play at col 2 or col 6 to block if P1 plays at 2 or 6.
    # Or P2 can play at 5 to make their own 3-in-a-row vertically.

    if test_board is None:
        logger.error("Failed to setup test board.")
    else:
        logger.info(f"Initial board state for MCTS test (Player {current_player}'s turn):\n"
                    f"{_convert_board_to_2D_string(test_board, game_config)}")

        # Setup model and device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        model_nn = ConnectXNet(n_actions=game_config['columns']).to(device)
        # In a real scenario, load a trained model:
        # model_nn = load_model(model_nn, path="./models/best", filename="best_model.pth", device=device)
        model_nn.eval()

        # RNG for MCTS (for Dirichlet noise and action selection)
        # In real use, this np_rng would be initialized and passed from train.py
        master_seed = 12345 
        process_id = 0 # Dummy process id for testing
        seed_for_rng = master_seed + process_id
        numpy_rng = np.random.default_rng(seed_for_rng)


        # Test select_mcts_action
        # Player 1 has just played at col 5. So it's player 2's turn next.
        current_player_for_mcts = 2 # P2 to play next
        
        logger.info(f"Testing MCTS for Player {current_player_for_mcts}...")
        
        # Make P1 place one more stone to create an immediate threat for P2
        # test_board_threatening = _apply_action(test_board, 6, 1, game_config) # P1 makes 4 in a row, game over
        # logger.info(f"Board if P1 plays at 6 (P1 wins):\n{_convert_board_to_2D_string(test_board_threatening, game_config)}")
        # is_done_threat, winner_threat = get_terminal_state_and_winner(test_board_threatening, game_config)
        # logger.info(f"Is terminal: {is_done_threat}, Winner: {winner_threat}")


        chosen_action, policy = select_mcts_action(
            initial_board=test_board.copy(), # Pass a copy
            initial_player_to_act=current_player_for_mcts,
            config=game_config,
            model=model_nn,
            n_simulations=100, # Reduced for quick test
            c_puct=1.5,
            c_fpu=0.5, # Example FPU
            device=device,
            np_rng=numpy_rng,
            dirichlet_alpha=0.3, # Example values
            dirichlet_epsilon=0.25,
            temperature=0.1, # Near-deterministic for testing best move
            log_debug=True      # Enable detailed MCTS logging
        )

        logger.info(f"MCTS selected action: {chosen_action}")
        logger.info(f"Policy (pi) from MCTS: {policy}")

        if chosen_action is not None:
            final_board = _apply_action(test_board, chosen_action, current_player_for_mcts, game_config)
            logger.info(f"Board after P{current_player_for_mcts} plays at {chosen_action}:\n"
                        f"{_convert_board_to_2D_string(final_board, game_config)}")
            is_done_final, winner_final = get_terminal_state_and_winner(final_board, game_config)
            logger.info(f"Game state after action: Done={is_done_final}, Winner={winner_final}")

        # Test terminal condition directly
        # P1: (0,3), (1,3), (2,3)
        # P2: (0,4), (1,4), (2,4)
        # P1 to play at (3,3) for win
        board_almost_win = np.zeros((6,7), dtype=int)
        board_almost_win[5,3]=1; board_almost_win[5,4]=2;
        board_almost_win[4,3]=1; board_almost_win[4,4]=2;
        board_almost_win[3,3]=1; board_almost_win[3,4]=2; 
        # P1 to play column 3 for win
        logger.info(f"Board almost win for P1:\n{_convert_board_to_2D_string(board_almost_win, game_config)}")
        
        action_p1_wins = 3
        board_p1_wins = _apply_action(board_almost_win.copy(), action_p1_wins, 1, game_config)
        is_done_p1w, winner_p1w = get_terminal_state_and_winner(board_p1_wins, game_config)
        logger.info(f"P1 plays at {action_p1_wins}. Done={is_done_p1w}, Winner={winner_p1w} (Expected True, 1)")
        logger.info(f"Board:\n{_convert_board_to_2D_string(board_p1_wins, game_config)}")

        # Test draw condition
        board_draw_almost = np.array([
            [1,2,1,2,1,2,1],
            [1,2,1,2,1,2,1],
            [2,1,2,1,2,1,2],
            [2,1,2,1,2,1,2],
            [1,2,1,2,1,2,1],
            [0,1,2,1,2,1,2] # P2 to play at col 0 for draw
        ], dtype=int)
        logger.info(f"Board almost draw for P2:\n{_convert_board_to_2D_string(board_draw_almost, game_config)}")
        board_draw = _apply_action(board_draw_almost.copy(), 0, 2, game_config)
        is_done_draw, winner_draw = get_terminal_state_and_winner(board_draw, game_config)
        logger.info(f"P2 plays at 0. Done={is_done_draw}, Winner={winner_draw} (Expected True, 0)")