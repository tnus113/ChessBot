import flask
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
import chess
import os
import random
import math
import numpy as np
import logging

# Tắt log không cần thiết của thư viện chess
logging.getLogger("chess").setLevel(logging.CRITICAL)

# Import các tiện ích từ chess_utils
from chess_utils import PolicyNetwork, ValueNetwork, board_to_tensor, encode_move

app = Flask(__name__, template_folder='templates')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. KHỞI TẠO VÀ TẢI MODEL
# ==========================================
print(f">>> Server đang chạy trên thiết bị: {DEVICE}")

# Tải Policy Network (Ưu tiên PPO -> SFT)
policy_net = PolicyNetwork().to(DEVICE)
if os.path.exists("ppo_chess_bot.pth"):
    try:
        policy_net.load_state_dict(torch.load("ppo_chess_bot.pth", map_location=DEVICE))
        print(" -> Policy: PPO (RLHF)")
    except:
        print(" -> Cảnh báo: Lỗi tải PPO, đang thử SFT...")
        if os.path.exists("sft_policy.pth"):
            policy_net.load_state_dict(torch.load("sft_policy.pth", map_location=DEVICE))
elif os.path.exists("sft_policy.pth"):
    policy_net.load_state_dict(torch.load("sft_policy.pth", map_location=DEVICE))
    print(" -> Policy: SFT (Supervised)")
else:
    print(" -> CẢNH BÁO: Không tìm thấy Policy Model nào!")

policy_net.eval()

# Tải Value Network (Reward Model)
value_net = ValueNetwork().to(DEVICE)
has_value_net = False
if os.path.exists("reward_model.pth"):
    try:
        value_net.load_state_dict(torch.load("reward_model.pth", map_location=DEVICE))
        print(" -> Value: Reward Model")
        has_value_net = True
    except:
        print(" -> Lỗi tải Reward Model.")
value_net.eval()

# ==========================================
# 2. THUẬT TOÁN MCTS (MONTE CARLO TREE SEARCH)
# ==========================================

class MCTSNode:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.prior = prior

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
            q_val = -child.value() # Đảo dấu (Zero-sum game)
            u_val = c_puct * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            
            score = q_val + u_val
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        
        return best_move, best_child

def mcts_search(board, simulations=400):
    """Chạy mô phỏng để tìm nước đi tối ưu"""
    root = MCTSNode()
    
    # 1. Mở rộng Root
    tensor = torch.from_numpy(board_to_tensor(board)).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = policy_net(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        idx = encode_move(move)
        if idx < 4096:
            root.children[move] = MCTSNode(parent=root, prior=probs[idx])
    
    # 2. Vòng lặp Mô phỏng
    for _ in range(simulations):
        node = root
        scratch_board = board.copy()
        
        # A. Select
        while node.is_expanded():
            move, node = node.select()
            scratch_board.push(move)
            
        # B. Evaluate & Expand
        value = 0
        if scratch_board.is_game_over():
            value = -1 if scratch_board.is_checkmate() else 0
        else:
            t_input = torch.from_numpy(board_to_tensor(scratch_board)).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                # Policy cho bước tiếp theo
                p_logits = policy_net(t_input)
                p_probs = F.softmax(p_logits, dim=1).cpu().numpy()[0]
                
                # Value đánh giá thế cờ (ép về [-1, 1])
                if has_value_net:
                    v_raw = value_net(t_input)
                    value = torch.tanh(v_raw).item()
                else:
                    value = 0

            # Mở rộng node lá
            l_moves = list(scratch_board.legal_moves)
            for m in l_moves:
                idx = encode_move(m)
                if idx < 4096:
                    node.children[m] = MCTSNode(parent=node, prior=p_probs[idx])
        
        # C. Backpropagate
        while node is not None:
            node.visits += 1
            node.value_sum += value
            value = -value
            node = node.parent

    # 3. Chọn kết quả
    if not root.children:
        return random.choice(legal_moves)
        
    sorted_moves = sorted(root.children.items(), key=lambda item: item[1].visits, reverse=True)
    top_move, top_node = sorted_moves[0]
    
    # Log thông tin tư duy
    win_rate = (-top_node.value() + 1) / 2 * 100
    print(f"Bot: {top_move.uci()} | WinRate: {win_rate:.1f}% | Visits: {top_node.visits}")
    
    return top_move

# ==========================================
# 3. API ROUTES
# ==========================================

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

        # Chạy MCTS (400 simulations là mức cân bằng tốt giữa tốc độ và sức mạnh)
        best_move = mcts_search(board, simulations=1600)
        
        return jsonify({'move': best_move.uci(), 'game_over': False})

    except Exception as e:
        print(f"Lỗi Server: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(">>> Server đã sẵn sàng tại http://127.0.0.1:5000")
    app.run(debug=True, port=5000)