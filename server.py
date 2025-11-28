import flask
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
import chess
import os
import random
import math
import numpy as np

# Import utils
from chess_utils import PolicyNetwork, ValueNetwork, board_to_tensor, encode_move

app = Flask(__name__, template_folder='templates')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. LOAD MODELS ---
print(">>> Đang tải các mô hình AI...")

# A. Policy Network (Trực giác - Bot PPO)
policy_net = PolicyNetwork().to(DEVICE)
if os.path.exists("ppo_chess_bot.pth"):
    try:
        policy_net.load_state_dict(torch.load("ppo_chess_bot.pth", map_location=DEVICE))
        print(" -> Đã tải Policy: PPO (Thông minh nhất)")
    except:
        print(" -> Lỗi tải PPO, thử dùng SFT...")
        if os.path.exists("sft_policy.pth"):
            policy_net.load_state_dict(torch.load("sft_policy.pth", map_location=DEVICE))
else:
    if os.path.exists("sft_policy.pth"):
        policy_net.load_state_dict(torch.load("sft_policy.pth", map_location=DEVICE))
        print(" -> Đã tải Policy: SFT (Học vẹt)")
    else:
        print(" -> CẢNH BÁO: Không có Policy Model!")

policy_net.eval()

# B. Value Network (Đánh giá - Reward Model)
value_net = ValueNetwork().to(DEVICE)
has_value_net = False
if os.path.exists("reward_model.pth"):
    try:
        value_net.load_state_dict(torch.load("reward_model.pth", map_location=DEVICE))
        print(" -> Đã tải Value: Reward Model")
        has_value_net = True
    except:
        print(" -> Lỗi tải Reward Model.")
value_net.eval()

# --- 2. THUẬT TOÁN MCTS (ALPHAZERO LOGIC) ---
class MCTSNode:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.prior = prior # Xác suất từ Policy

    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visits == 0: return 0
        return self.value_sum / self.visits

    def select(self, c_puct=1.0):
        """Chọn node con tốt nhất theo công thức PUCT"""
        best_score = -float('inf')
        best_move = None
        best_child = None

        for move, child in self.children.items():
            # UCB = Q + U
            # Q: Giá trị trung bình (đảo dấu vì đối thủ sẽ chọn nước hại mình)
            q_val = -child.value()
            
            # U: Tiềm năng khai phá dựa trên Prior và số lần thăm
            u_val = c_puct * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            
            score = q_val + u_val
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        
        return best_move, best_child

def mcts_search(board, simulations=400):
    root = MCTSNode()
    
    # 1. Expand Root
    tensor = torch.from_numpy(board_to_tensor(board)).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = policy_net(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    
    legal_moves = list(board.legal_moves)
    # Tạo các node con cho root
    for move in legal_moves:
        idx = encode_move(move)
        if idx < 4096:
            root.children[move] = MCTSNode(parent=root, prior=probs[idx])
    
    # 2. Simulation Loop
    for _ in range(simulations):
        node = root
        scratch_board = board.copy()
        
        # A. Select (Đi xuống lá)
        while node.is_expanded():
            move, node = node.select()
            scratch_board.push(move)
            
        # B. Evaluate & Expand (Tại node lá)
        value = 0
        if scratch_board.is_game_over():
            if scratch_board.is_checkmate():
                value = -1 # Bị chiếu hết là thua (-1)
            else:
                value = 0 # Hòa
        else:
            # Dùng Value Network (RM) để chấm điểm thế cờ
            t_input = torch.from_numpy(board_to_tensor(scratch_board)).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                # Policy head để mở rộng
                p_logits = policy_net(t_input)
                p_probs = F.softmax(p_logits, dim=1).cpu().numpy()[0]
                
                # Value head để đánh giá
                if has_value_net:
                    v_raw = value_net(t_input)
                    # Ép về [-1, 1] vì ValueNetwork hiện tại ko có Tanh
                    value = torch.tanh(v_raw).item()
                else:
                    value = 0 # Nếu ko có Value Net thì coi như hòa (hoặc dùng random rollout)

            # Mở rộng node lá
            l_moves = list(scratch_board.legal_moves)
            for m in l_moves:
                idx = encode_move(m)
                if idx < 4096:
                    node.children[m] = MCTSNode(parent=node, prior=p_probs[idx])
        
        # C. Backpropagate (Lan truyền ngược)
        while node is not None:
            node.visits += 1
            node.value_sum += value
            value = -value # Đổi phe (Negamax)
            node = node.parent

    # 3. Chọn nước đi tốt nhất (dựa trên số lần thăm - Robustness)
    if not root.children:
        return random.choice(legal_moves)
        
    # Sắp xếp các nước đi theo số lần thăm giảm dần
    sorted_moves = sorted(root.children.items(), key=lambda item: item[1].visits, reverse=True)
    
    # Debug: In ra top 3 nước đi Bot suy nghĩ
    print(f"\n--- Bot Thinking (Sims: {simulations}) ---")
    for m, node in sorted_moves[:3]:
        score = -node.value() # Điểm từ góc nhìn của Bot
        print(f"Move: {m.uci()} | Visits: {node.visits} | Score: {score:.3f}")
        
    return sorted_moves[0][0]

# --- 3. SERVER ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_move', methods=['POST'])
def get_move():
    try:
        data = request.json
        fen = data.get('fen')
        board = chess.Board(fen)
        
        if board.is_game_over():
            return jsonify({'move': None, 'game_over': True})

        # CHẠY MCTS
        # Số simulations càng cao bot càng mạnh nhưng càng chậm
        # 400 là mức chuẩn của AlphaGo Zero (bản nhẹ)
        best_move = mcts_search(board, simulations=400)
        
        return jsonify({'move': best_move.uci(), 'game_over': False})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("MCTS Chess Server Running...")
    app.run(debug=True, port=5000)