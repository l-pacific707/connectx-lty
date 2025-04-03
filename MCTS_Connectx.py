# MCTS 기반 AlphaZero-style 노드 구조 및 탐색 설계
from logger_setup import get_logger
import math
import numpy as np
from collections import defaultdict
import copy
import torch
import ConectXNN as cxnn


logger = get_logger("MCTS","MCTS.log")


class MCTSNode:
    def __init__(self, state, parent=None, prior=0.0):
        self.state = state #envrionment 전체가 사용될 예정
        self.parent = parent
        self.children = {}
        if parent is not None:
            temp = [2,1]
            self.current_player = temp[parent.current_player - 1] #parent 는 current player를 가지고 있다. 왜? 이전에 선택을 받았으니까.
        else:
            self.current_player = self.find_current_player()

        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = prior

    def is_expanded(self):
        return len(self.children) > 0

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = MCTSNode(state=None, parent=self, prior=prob)

    def select_child(self, c_puct):
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in self.children.items():
            u = c_puct * child.P * math.sqrt(self.N) / (1 + child.N)
            score = child.Q + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def backup(self, value):
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

        if self.parent:
            self.parent.backup(-value)
            
    def find_current_player(self) -> int:
        if self.state.state[0]["status"] == "ACTIVE":
            return 1
        elif self.state.state[1]["status"] == "ACTIVE":
            return 2
        else:
            logger.debug(f"current state is terminal state of the game. State: {self.state.state}")
            if self.parent.current_player == 1:
                return 2
            elif self.parent.current_player == 2:
                return 1
            else:
                logger.error("parent node had current_player value other than 1 or 2.")
                raise ValueError
            

def simulate_env(env, action):
    env_copy = copy.deepcopy(env)
    env_copy.step([action, action])  # 상대 action은 무시됨
    return env_copy

def is_terminal(env):
    return env.done  # Kaggle env는 .done 속성으로 게임 종료 여부 제공

def make_tree(root_env, model, n_simulations, c_puct):
    root_node = MCTSNode(state=root_env)
    logger.debug(f"Start to make MC tree. \n -----root state-----\n{convert_board_to_2D(root_env)} ------ \ncurrent player : {root_node.current_player}  ")

    input = cxnn.preprocess_input(root_env)
    p_logits, v = model(input)
    p = torch.softmax(p_logits, dim=-1).detach().cpu().numpy().flatten()
    valid_actions = get_valid_actions(root_env)
    priors = [(a, p[a]) for a in valid_actions]
    root_node.expand(priors)
    root_node.backup(v.item())
    logger.debug(f"Root node value evaluated as : {v.item()}")

    for _ in range(n_simulations):
        node = root_node
        env = root_env

        # Selection + Simulation
        while node.is_expanded():
            logger.debug(f"current node is already expanded. Start to select child")
            action, node = node.select_child(c_puct)
            logger.debug(f"selected action : {action}")
            env = simulate_env(env, action)
            

            # 선택이 되기전에는 node.state = None이다가, 선택이 되고나면 그제서야 state를 저장.
            if node.state is None:
                node.state = env

            if is_terminal(env):
                #logger.debug(f"selected child state is terminal state of the game. \n------root node's state------ \n{convert_board_to_2D(root_env.state[0]["observation"]["board"])}. \n Terminal state : {env.state.state[0]["observation"]["board"]}\n initial player : {root_node.current_player}, final player : {node.current_player} ")
                logger.debug(f"type of env: {type(env)}")
                break
        #leaf node 에 도달 혹은 게임이 끝남.
        if is_terminal(env):
            # 게임이 끝났다면, 승패 결과 z를 구해서 backup
            # 근데 current_player 를 알아야함.
            z = get_game_result(node)  # +1, -1, 0 형태로 반환되어야 함
            node.backup(z)
            continue

        # Leaf Evaluation
        logger.debug(f"Reached to leaf node. start to evaluate p, v for leaf node. and Expand this leaf node.")
        input = cxnn.preprocess_input(env)
        p_logits, v = model(input)
        p = torch.softmax(p_logits, dim=-1).detach().cpu().numpy().flatten()
        valid_actions = get_valid_actions(env)
        logger.debug(f" total expanded actions : {len(valid_actions)}")
        priors = [(a, p[a]) for a in valid_actions]

        node.expand(priors)
        node.backup(v.item())

    return root_node

def create_pi(root_node, num_actions, temperature=1.0):
    pi = np.zeros(num_actions)
    visit_counts = np.zeros(num_actions)

    for action, child in root_node.children.items():
        visit_counts[action] = child.N

    if temperature == 0:
        best_action = np.argmax(visit_counts)
        pi[best_action] = 1.0
    else:
        counts_pow = visit_counts ** (1.0 / temperature)
        if counts_pow.sum() == 0:
            pi = np.ones(num_actions) / num_actions
        else:
            pi = counts_pow / counts_pow.sum()

    return pi

def get_observation(env):
    """The output of this function will go to model(env). 
    specifically designed for connectX connectX environment

    Args:
        env (connectX environment): _description_
    
    Returns:
        obs(dict)
    """
    obs = env.state[0]["observation"]
    return obs

def get_valid_actions(env):
    """Specially designed for ConnectX env.

    Args:
        env (connectx environment): _description_
    """
    board_width = env.configuration.columns
    board_height = env.configuration.rows
    board = np.array(env.state[0]["observation"]["board"]).reshape((board_height, board_width))
 
    valid_actions = [c for c in range(board_width) if board[0][c] == 0]
    return valid_actions
    ...

def get_game_result(node):
    if node.state.done:
        result = node.state.state[node.current_player - 1]["reward"] # -1 to make index
        if isinstance(result, int):
            return result 
        else:
            raise ValueError("game result was not integer.")
    else:
        logger.error("Getting result of game without being terminated.")
        raise ValueError

def convert_board_to_2D(env):
    board = np.array(env.state[0]["observation"]["board"]).reshape((6, 7))
    return board

def select_action(root_env, model, n_simulations, c_puct, temperature=1.0):
    root_node = make_tree(root_env, model, n_simulations, c_puct)
    pi = create_pi(root_node, root_env.configuration.columns, temperature)
    action = np.random.choice(root_env.configuration.columns, p=pi)
    return action , pi
    
if __name__ == "__main__":
    from kaggle_environments import make
    import ConectXNN as cxnn
    env = make("connectx")
    env.reset()
    for i in range(4):
        env.step([i,i])
    model = cxnn.ConnectXNet()
    root_node = make_tree(env, model, n_simulations=10, c_puct=1.0)