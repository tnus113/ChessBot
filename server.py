import flask
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np
import os

# Khởi tạo Flask App
app = Flask(__name__, template_folder='templates')

# --- 1. ĐỊNH NGHĨA MODEL & HÀM XỬ LÝ (Giống lúc Train) ---

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        # Input: 12 channels (6 loại quân x 2 màu)
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 4096) # Output 4096 nước đi

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(-1, 128 * 8 * 8) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def board_to_tensor(board):
    """Chuyển đổi bàn cờ sang Tensor 12x8x8"""
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    layers = []
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in pieces:
            layer = np.zeros((8, 8), dtype=np.float32)
            for square in board.pieces(piece_type, color):
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                layer[rank][file] = 1.0
            layers.append(layer)
    return np.array(layers)

def encode_move(move):
    """Chuyển đổi nước đi sang index (0-4095)"""
    return move.from_square * 64 + move.to_square

# --- 2. TẢI MODEL ĐÃ TRAIN ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessNet().to(DEVICE)

# Ưu tiên load model tốt nhất (best), nếu không có thì load bản thường
if os.path.exists("best_chess_bot.pth"):
    MODEL_PATH = "best_chess_bot.pth"
elif os.path.exists("chess_puzzle_bot.pth"):
    MODEL_PATH = "chess_puzzle_bot.pth"
else:
    MODEL_PATH = None

if MODEL_PATH:
    try:
        # map_location để đảm bảo chạy được cả trên máy không có GPU
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval() # Chuyển sang chế độ dự đoán (tắt Dropout/BatchNorm train)
        print(f">>> Đã load Model thành công từ: {MODEL_PATH}")
    except Exception as e:
        print(f"!!! Lỗi khi load model: {e}")
        MODEL_PATH = None
else:
    print("!!! CẢNH BÁO: Không tìm thấy file model (.pth). Bot sẽ đánh random.")

# --- 3. API ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_move', methods=['POST'])
def get_move():
    try:
        data = request.json
        fen = data.get('fen')
        board = chess.Board(fen)
        
        # Kiểm tra xem game đã hết chưa
        if board.is_game_over():
            return jsonify({'move': None, 'game_over': True, 'result': board.result()})

        # --- LOGIC CHỌN NƯỚC ĐI THÔNG MINH (MASKING) ---
        
        # 1. Lấy danh sách TẤT CẢ các nước đi hợp lệ
        legal_moves = list(board.legal_moves)
        
        # Nếu không còn nước đi nào (hết cờ hoặc hết nước)
        if not legal_moves:
            return jsonify({'move': None, 'game_over': True})

        best_move = None
        
        # Nếu có Model, dùng Model để chọn nước tốt nhất trong số các nước hợp lệ
        if MODEL_PATH:
            # Chuyển bàn cờ thành Tensor
            tensor = torch.from_numpy(board_to_tensor(board)).float().unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                # Model chấm điểm cho 4096 khả năng
                outputs = model(tensor) # Shape [1, 4096]
                
                # Tìm nước đi hợp lệ có điểm cao nhất
                best_score = -float('inf')
                
                for move in legal_moves:
                    move_idx = encode_move(move)
                    
                    # Kiểm tra an toàn: chỉ xét các nước đi nằm trong phạm vi Model học (0-4095)
                    # Nước phong cấp (Promotion) có thể có index khác, tạm thời bỏ qua để đơn giản
                    if move_idx < 4096:
                        score = outputs[0][move_idx].item()
                        
                        if score > best_score:
                            best_score = score
                            best_move = move
                
                if best_move:
                    print(f"Bot chọn: {best_move} (Score: {best_score:.4f})")
                else:
                    # Trường hợp hiếm: Chỉ còn các nước phong cấp phức tạp mà bot chưa học
                    print("Bot dùng fallback random (do nước đi đặc biệt).")
                    import random
                    best_move = random.choice(legal_moves)

        else:
            # Nếu không có Model, đánh random
            import random
            best_move = random.choice(legal_moves)
            print("Bot đánh random (No model).")

        return jsonify({'move': best_move.uci(), 'game_over': False})

    except Exception as e:
        print(f"Lỗi Server: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Chạy server
    print("Server đang chạy tại http://127.0.0.1:5000")
    app.run(debug=True, port=5000)