import train as tr
import ConnectXNN as cxnn
import numpy as np
import torch
from kaggle_environments import make, evaluate

model1 = cxnn.ConnectXNet()
model1 = cxnn.load_model(model1, './models/best', 'best_model.pth')
model2 = cxnn.ConnectXNet()
model2 = cxnn.load_model(model2, './models/checkpoints', 'model_iter_60.pth')

def my_agent1(observation, configuration):
    mark = observation.mark
    row = configuration.rows
    col = configuration.columns
    
    board = np.array(observation.board).reshape(row, col)
    
    P1 = (board == 1).astype(np.float32)
    P2 = ((board == 2)).astype(np.float32)
    if mark == 2:
        player_plane = np.full((row, col), 1, dtype=np.float32)
    elif mark == 1:
        player_plane = np.full((row, col), -1, dtype=np.float32)
    
    input = torch.tensor(np.stack([P1, P2, player_plane])).unsqueeze(0)
    
    p_logits, v = model1(input)
    p_logits = p_logits.detach().cpu().numpy().flatten()
    
    valid_actions = [c for c in range(col) if board[0][c] == 0]
    mask = np.full_like(p_logits, -np.inf)
    mask[valid_actions] = p_logits[valid_actions]
    
    # 필터링된 logits에 softmax 적용
    filtered_probs = np.exp(mask) / np.sum(np.exp(mask))
    
    action = np.random.choice(len(filtered_probs), p=filtered_probs)
    
        
    return action

def my_agent2(observation, configuration):
    mark = observation.mark
    row = configuration.rows
    col = configuration.columns
    
    board = np.array(observation.board).reshape(row, col)
    
    P1 = (board == 1).astype(np.float32)
    P2 = ((board == 2)).astype(np.float32)
    if mark == 2:
        player_plane = np.full((row, col), 1, dtype=np.float32)
    elif mark == 1:
        player_plane = np.full((row, col), -1, dtype=np.float32)
    
    input = torch.tensor(np.stack([P1, P2, player_plane])).unsqueeze(0)
    
    p_logits, v = model2(input)
    p_logits = p_logits.detach().cpu().numpy().flatten()
    
    valid_actions = [c for c in range(col) if board[0][c] == 0]
    mask = np.full_like(p_logits, -np.inf)
    mask[valid_actions] = p_logits[valid_actions]
    
    # 필터링된 logits에 softmax 적용
    filtered_probs = np.exp(mask) / np.sum(np.exp(mask))
    
    action = np.random.choice(len(filtered_probs), p=filtered_probs)
    
        
    return action

if __name__ == "__main__":
    env = make("connectx")
    env.reset()
    env.run([my_agent1, my_agent2])
    print("Game finished.")
    print(env.render(mode="ansi"))