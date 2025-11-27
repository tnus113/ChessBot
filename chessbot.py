import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch.nn.functional as F

# Kiểm tra GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

# --- 1. HÀM CHUYỂN BÀN CỜ THÀNH TENSOR ---
def board_to_tensor(board):
    # 12 lớp: 6 loại quân (Trắng) + 6 loại quân (Đen)
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    layers = []
    
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in pieces:
            layer = np.zeros((8, 8), dtype=np.float32)
            # Lấy vị trí các quân cờ
            for square in board.pieces(piece_type, color):
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                layer[rank][file] = 1.0
            layers.append(layer)
            
    return np.array(layers) # Shape: (12, 8, 8)

# --- 2. HÀM MÃ HÓA NƯỚC ĐI ---
def encode_move(move):
    # Đơn giản hóa: from_square * 64 + to_square (Tổng 4096 output)
    return move.from_square * 64 + move.to_square

def decode_move(index):
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq)

class HFChessDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Lấy một mẫu dữ liệu từ Hugging Face dataset
        item = self.dataset[idx]
        
        # Cấu trúc item của 'Lichess/chess-puzzles' thường là:
        # {'FEN': '...', 'Moves': 'e2e4 c7c5 ...', 'Rating': ...}
        
        fen = item['FEN']
        moves_str = item['Moves']
        
        board = chess.Board(fen)
        moves = moves_str.split()
        
        # LOGIC QUAN TRỌNG:
        # Puzzles thường bắt đầu bằng nước đi của đối thủ (nước sai lầm hoặc dẫn nhập)
        # Nước đi đầu tiên trong chuỗi 'Moves' là nước đối thủ vừa đi
        opponent_move = chess.Move.from_uci(moves[0])
        board.push(opponent_move)
        
        # Bây giờ đến lượt Bot đi. Đây là trạng thái đầu vào (Input)
        x_tensor = torch.from_numpy(board_to_tensor(board)).float()
        
        # Nước đi thứ 2 trong chuỗi là đáp án đúng (Label)
        # Bot cần học để dự đoán nước này
        target_move = chess.Move.from_uci(moves[1])
        y_label = encode_move(target_move)
        
        return x_tensor, y_label
    
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        # Input: 12 channels -> Output: 64
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 4096) # 4096 class nước đi

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(-1, 128 * 8 * 8) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Output logits
        return x
    
def train():
    # 1. LOAD DATASET TỪ HUGGING FACE
    print("Đang tải dataset từ Hugging Face...")
    # Lấy 20k mẫu để train demo. Bỏ "[:20000]" nếu muốn train hết (rất lâu)
    hf_dataset = load_dataset("Lichess/chess-puzzles", split="train[:200000]")
    
    # Tạo Wrapper Dataset và DataLoader
    train_dataset = HFChessDataset(hf_dataset)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    
    print(f"Đã tải xong: {len(train_dataset)} mẫu puzzles.")

    # 2. KHỞI TẠO MODEL
    model = ChessNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 3. TRAINING LOOP
    EPOCHS = 5
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for batch_idx, (boards, labels) in enumerate(train_loader):
            boards, labels = boards.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(boards) # Forward
            
            loss = criterion(outputs, labels) # Calc Loss
            loss.backward() # Backward
            optimizer.step() # Update weights
            
            total_loss += loss.item()
            
            # Tính độ chính xác (Accuracy) sơ bộ
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total_samples
        print(f"=== KẾT THÚC EPOCH {epoch+1} | Avg Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}% ===")

    # 4. LƯU MODEL
    torch.save(model.state_dict(), "chess_puzzle_bot.pth")
    print("Đã lưu model thành công!")

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    train()