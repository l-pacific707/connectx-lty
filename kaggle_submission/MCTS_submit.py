# MCTS_Connectx_modified.py
import math
import numpy as np
import copy
import torch
import torch.nn.functional as F

# Assuming ConnectXNN is in the same directory or accessible in PYTHONPATH
# For MCTS internal use, we'll define a helper to create NN input
# import ConnectXNN as cxnn # This would be used by the training script, not directly in MCTS logic if env is removed

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
        
        for action, child in self.children.items():
            if child.N == 0: 
                q_term_child_perspective = 0.0 
                if c_fpu > 0: 
                    pass 
            else:
                q_term_child_perspective = child.W / child.N 

            q_value_for_parent = -q_term_child_perspective 
            
            u_value = c_puct * child.P * sqrt_total_N_parent / (1 + child.N)
            score = q_value_for_parent + u_value

            # Removed empty 'if log_debug:' block here

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        if best_child is None:
            if self.children: 
                best_action = list(self.children.keys())[0]
                best_child = self.children[best_action]
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
            node.W += current_perspective_value
            node.V = node.W / node.N 

            current_perspective_value = -current_perspective_value 
            node = node.parent


def get_game_result_for_player(board, config, perspective_player):
    """
    Determine the game result (+1 win, -1 loss, 0 draw) from the perspective
    of the given player.
    """
    is_done, winner = get_terminal_state_and_winner(board, config)

    if not is_done:
        return 0.0 

    if winner == perspective_player:
        return 1.0
    elif winner == 0: # Draw
        return 0.0
    elif winner == -1: 
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

    valid_actions_root = _get_valid_actions(initial_board, config)
    if not valid_actions_root:
        return root_node 

    try:
        with torch.no_grad():
            nn_input_tensor = _create_nn_input(initial_board, initial_player_to_act, config).to(device)
            p_logits, v_nn = model(nn_input_tensor) 
            policy_priors_all = torch.softmax(p_logits, dim=-1).squeeze(0).cpu().numpy() 
            value_estimate_for_root_player = v_nn.item() 

        action_priors_valid = []
        raw_priors_for_noise = []
        for action in valid_actions_root:
            if 0 <= action < len(policy_priors_all):
                raw_priors_for_noise.append(policy_priors_all[action])
                action_priors_valid.append((action, policy_priors_all[action])) 


        if dirichlet_alpha > 0 and len(raw_priors_for_noise) > 0:
            noise = np_rng.dirichlet([dirichlet_alpha] * len(raw_priors_for_noise))
            
            final_action_priors = []
            for i, (action, raw_prior) in enumerate(action_priors_valid):
                noisy_prior = (1 - dirichlet_epsilon) * raw_prior + dirichlet_epsilon * noise[i]
                final_action_priors.append((action, noisy_prior))
        else:
            final_action_priors = action_priors_valid

        root_node.N = 1 
        root_node.W = value_estimate_for_root_player 
        root_node.V = value_estimate_for_root_player

        root_node.expand(final_action_priors)

    except Exception as e:
        # print(f"Error during root node evaluation/expansion: {e}") # Optional: print error
        return root_node

    for sim_idx in range(n_simulations):
        current_node = root_node
        sim_board = initial_board.copy()

        while current_node.is_expanded():
            action, next_node_template = current_node.select_child(c_puct, c_fpu, log_debug)

            if action is None or next_node_template is None:
                current_node = None 
                break

            new_sim_board = _apply_action(sim_board, action, current_node.player_to_act, config)
            if new_sim_board is None:
                current_node = None 
                break
            sim_board = new_sim_board
            
            current_node = next_node_template 
            
            if current_node.board is None:
                current_node.board = sim_board 

            is_terminal_leaf, _ = get_terminal_state_and_winner(sim_board, config)
            if is_terminal_leaf:
                break
        
        if current_node is None: continue 

        leaf_node = current_node
        value_for_leaf_player = 0.0

        is_terminal_final, winner_final = get_terminal_state_and_winner(leaf_node.board, config)

        if is_terminal_final:
            value_for_leaf_player = get_game_result_for_player(leaf_node.board, config, leaf_node.player_to_act)
        else:
            try:
                with torch.no_grad():
                    nn_input_tensor = _create_nn_input(leaf_node.board, leaf_node.player_to_act, config).to(device)
                    p_logits_leaf, v_nn_leaf = model(nn_input_tensor)
                    policy_priors_leaf_all = torch.softmax(p_logits_leaf, dim=-1).squeeze(0).cpu().numpy()
                    value_for_leaf_player = v_nn_leaf.item() 

                valid_actions_leaf = _get_valid_actions(leaf_node.board, config)
                if valid_actions_leaf:
                    leaf_action_priors = []
                    for act_leaf in valid_actions_leaf:
                         if 0 <= act_leaf < len(policy_priors_leaf_all):
                            leaf_action_priors.append((act_leaf, policy_priors_leaf_all[act_leaf]))
                    
                    leaf_node.expand(leaf_action_priors)
                else:
                    is_term_recheck, winner_recheck = get_terminal_state_and_winner(leaf_node.board, config)
                    if is_term_recheck:
                        value_for_leaf_player = get_game_result_for_player(leaf_node.board, config, leaf_node.player_to_act)
                        # Removed: if log_debug: logger.debug(...)
                    else:
                        # print(f"Sim {sim_idx + 1}: Leaf node has no valid actions but not terminal. Board:\n{_convert_board_to_2D_string(leaf_node.board, config)}") # Optional
                        value_for_leaf_player = 0.0 
            except Exception as e:
                # print(f"Error during leaf evaluation/expansion: {e}") # Optional
                continue 

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
        return pi

    # Note: The original MCTSNode(None,None,None) might be problematic if config is needed by a default node.
    # However, for just accessing .N (which is 0 for a default node), it might not crash immediately.
    # A safer default might be MCTSNode(None, None, root_node.config) if the default node could be used further.
    child_visit_counts = np.array([root_node.children.get(action, MCTSNode(None,None,root_node.config)).N 
                                  for action in range(num_actions)], dtype=np.float32)

    if np.sum(child_visit_counts) == 0: 
        num_children_actually_present = len(root_node.children)
        if num_children_actually_present > 0:
            uniform_prob = 1.0 / num_children_actually_present
            for action_key in root_node.children.keys():
                if 0 <= action_key < num_actions:
                    pi[action_key] = uniform_prob
        return pi


    if temperature == 0: 
        max_visits = -1
        best_action = -1
        for action_idx, child_node in root_node.children.items():
            if child_node.N > max_visits:
                max_visits = child_node.N
                best_action = action_idx
        
        if best_action != -1: 
            pi[best_action] = 1.0
        else: 
            pass # pi remains zeros
    else:
        powered_counts = child_visit_counts ** (1.0 / temperature)
        sum_powered_counts = np.sum(powered_counts)
        if sum_powered_counts > 1e-9:
            pi = powered_counts / sum_powered_counts
        else: 
            num_actual_children = len(root_node.children)
            if num_actual_children > 0:
                uniform_prob = 1.0 / num_actual_children
                for action_key in root_node.children.keys():
                     if 0 <= action_key < num_actions:
                          pi[action_key] = uniform_prob

    if np.isnan(pi).any() or np.isinf(pi).any():
        num_actual_children = len(root_node.children)
        if num_actual_children > 0:
            uniform_prob = 1.0 / num_actual_children
            pi = np.zeros(num_actions, dtype=np.float32)
            for action_key in root_node.children.keys():
                if 0 <= action_key < num_actions:
                    pi[action_key] = uniform_prob
        else:
            pi = np.zeros(num_actions, dtype=np.float32) 
    elif abs(np.sum(pi) - 1.0) > 1e-5 and np.sum(pi) > 1e-9 : 
         pi /= np.sum(pi)
    elif np.sum(pi) < 1e-9 and len(root_node.children) > 0 : 
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
    
    current_valid_actions = _get_valid_actions(initial_board, config)
    if not current_valid_actions:
        return None, np.zeros(num_actions, dtype=np.float32)

    root = make_tree(initial_board, initial_player_to_act, config,
                     model, n_simulations, c_puct, c_fpu, device,
                     np_rng, dirichlet_alpha, dirichlet_epsilon, log_debug)

    pi_policy = create_mcts_policy(root, temperature)

    if np.sum(pi_policy) < 1e-6:
        if not current_valid_actions: 
            return None, np.zeros(num_actions, dtype=np.float32)
            
        pi_policy = np.zeros(num_actions, dtype=np.float32)
        prob_uniform = 1.0 / len(current_valid_actions)
        for action_val in current_valid_actions:
            pi_policy[action_val] = prob_uniform
        
        chosen_action = np_rng.choice(current_valid_actions) 
        return chosen_action, pi_policy

    try:
        pi_normalized = pi_policy / np.sum(pi_policy)
        chosen_action = np_rng.choice(num_actions, p=pi_normalized)
    except ValueError as e: 
        if not current_valid_actions: return None, np.zeros(num_actions, dtype=np.float32)
        chosen_action = np_rng.choice(current_valid_actions)
        pi_policy = np.zeros(num_actions, dtype=np.float32)
        prob_uniform = 1.0 / len(current_valid_actions)
        for action_val in current_valid_actions:
            pi_policy[action_val] = prob_uniform
        return chosen_action, pi_policy

    if chosen_action not in current_valid_actions:
        highest_prob_valid_action = -1
        max_prob_for_valid = -1.0
        for valid_act in current_valid_actions:
            if pi_policy[valid_act] > max_prob_for_valid:
                max_prob_for_valid = pi_policy[valid_act]
                highest_prob_valid_action = valid_act
        
        if highest_prob_valid_action != -1:
            chosen_action = highest_prob_valid_action
        else: 
            if not current_valid_actions: return None, np.zeros(num_actions, dtype=np.float32) 
            chosen_action = np_rng.choice(current_valid_actions)

    return chosen_action, pi_policy


if __name__ == "__main__":
    # Assuming ConnectXNN.py is in the same directory or python path
    try:
        from ConnectXNN import ConnectXNet #, load_model # Assuming load_model is part of ConnectXNN or not needed for this dummy
    except ImportError:
        # Define a dummy model for testing MCTS logic if ConnectXNN is not available
        class DummyConnectXNet(torch.nn.Module):
            def __init__(self, input_channels=3, board_size=(6,7), n_actions=7, num_res_blocks=1): # Added reasonable defaults
                super().__init__()
                self.n_actions = n_actions
                # Simplified dummy structure
                self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
                self.fc_p = torch.nn.Linear(16 * board_size[0] * board_size[1], n_actions) 
                self.fc_v = torch.nn.Linear(16 * board_size[0] * board_size[1], 1)  

            def forward(self, x): # x shape (B, C, H, W)
                batch_size = x.shape[0]
                x = torch.relu(self.conv1(x))
                x = x.view(batch_size, -1) # Flatten
                policy_logits = self.fc_p(x) 
                value = torch.tanh(self.fc_v(x)) 
                return policy_logits, value
        ConnectXNet = DummyConnectXNet
        # def load_model(model, path, filename, device=None): return model # Dummy load, if needed

    
    game_config = {'rows': 6, 'columns': 7, 'inarow': 4}
    test_board = np.zeros((game_config['rows'], game_config['columns']), dtype=int)
    current_player = 1

    test_board = _apply_action(test_board, 3, 1, game_config) # P1
    if test_board is not None: test_board = _apply_action(test_board, 3, 2, game_config) # P2
    if test_board is not None: test_board = _apply_action(test_board, 4, 1, game_config) # P1
    if test_board is not None: test_board = _apply_action(test_board, 4, 2, game_config) # P2
    if test_board is not None: test_board = _apply_action(test_board, 5, 1, game_config) # P1

    if test_board is None:
        print("Failed to setup test board.")
    else:
        print(f"Initial board state for MCTS test (Player {current_player}'s turn):\n"
                    f"{_convert_board_to_2D_string(test_board, game_config)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Adjusted DummyConnectXNet instantiation to match expected parameters if original is not found
        model_nn = ConnectXNet(input_channels=3, board_size=(game_config['rows'],game_config['columns']), n_actions=game_config['columns']).to(device)
        model_nn.eval()

        master_seed = 12345 
        process_id = 0 
        seed_for_rng = master_seed + process_id
        numpy_rng = np.random.default_rng(seed_for_rng)

        current_player_for_mcts = 2 
        
        print(f"Testing MCTS for Player {current_player_for_mcts}...")
        
        chosen_action, policy = select_mcts_action(
            initial_board=test_board.copy(), 
            initial_player_to_act=current_player_for_mcts,
            config=game_config,
            model=model_nn,
            n_simulations=100, 
            c_puct=1.5,
            c_fpu=0.5, 
            device=device,
            np_rng=numpy_rng,
            dirichlet_alpha=0.3, 
            dirichlet_epsilon=0.25,
            temperature=0.1, 
            log_debug=False      # log_debug is False, so internal logging prints won't show
        )

        print(f"MCTS selected action: {chosen_action}")
        print(f"Policy (pi) from MCTS: {policy}")

        if chosen_action is not None:
            final_board = _apply_action(test_board, chosen_action, current_player_for_mcts, game_config)
            print(f"Board after P{current_player_for_mcts} plays at {chosen_action}:\n"
                        f"{_convert_board_to_2D_string(final_board, game_config)}")
            is_done_final, winner_final = get_terminal_state_and_winner(final_board, game_config)
            print(f"Game state after action: Done={is_done_final}, Winner={winner_final}")

        board_almost_win = np.zeros((6,7), dtype=int)
        board_almost_win[5,3]=1; board_almost_win[5,4]=2;
        board_almost_win[4,3]=1; board_almost_win[4,4]=2;
        board_almost_win[3,3]=1; board_almost_win[3,4]=2; 
        print(f"Board almost win for P1:\n{_convert_board_to_2D_string(board_almost_win, game_config)}")
        
        action_p1_wins = 3
        board_p1_wins = _apply_action(board_almost_win.copy(), action_p1_wins, 1, game_config)
        is_done_p1w, winner_p1w = get_terminal_state_and_winner(board_p1_wins, game_config)
        print(f"P1 plays at {action_p1_wins}. Done={is_done_p1w}, Winner={winner_p1w} (Expected True, 1)")
        print(f"Board:\n{_convert_board_to_2D_string(board_p1_wins, game_config)}")

        board_draw_almost = np.array([
            [1,2,1,2,1,2,1],
            [1,2,1,2,1,2,1],
            [2,1,2,1,2,1,2],
            [2,1,2,1,2,1,2],
            [1,2,1,2,1,2,1],
            [0,1,2,1,2,1,2] 
        ], dtype=int)
        print(f"Board almost draw for P2:\n{_convert_board_to_2D_string(board_draw_almost, game_config)}")
        board_draw = _apply_action(board_draw_almost.copy(), 0, 2, game_config)
        is_done_draw, winner_draw = get_terminal_state_and_winner(board_draw, game_config)
        print(f"P2 plays at 0. Done={is_done_draw}, Winner={winner_draw} (Expected True, 0)")