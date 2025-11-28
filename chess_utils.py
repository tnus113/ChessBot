import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- XỬ LÝ DỮ LIỆU ---
def board_to_tensor(board):
    """
    Chuyển đổi bàn cờ sang Tensor 14x8x8:
    - 0-11: Vị trí quân (P, N, B, R, Q, K) x (Trắng, Đen)
    - 12: Lượt đi (Toàn số 1 nếu là lượt mình, 0 nếu lượt địch)
    - 13: Quyền nhập thành (Castling rights)
    """
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    layers = np.zeros((14, 8, 8), dtype=np.float32)
    
    # 1. Vị trí quân (Channel 0-11)
    for i, color in enumerate([chess.WHITE, chess.BLACK]):
        for j, piece_type in enumerate(pieces):
            idx = i * 6 + j
            for square in board.pieces(piece_type, color):
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                layers[idx][rank][file] = 1.0

    # 2. Lượt đi (Channel 12)
    if board.turn == chess.WHITE:
        layers[12, :, :] = 1.0

    # 3. Quyền nhập thành (Channel 13)
    if board.has_castling_rights(board.turn):
        layers[13, :, :] = 1.0
            
    return layers

def encode_move(move):
    return move.from_square * 64 + move.to_square

# --- KIẾN TRÚC MẠNG NƠ-RON (RESNET) ---

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ChessResNet(nn.Module):
    """Backbone ResNet mạnh mẽ hơn"""
    def __init__(self, num_res_blocks=4):
        super(ChessResNet, self).__init__()
        
        # Input layer (14 channels input)
        self.conv_input = nn.Conv2d(14, 128, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(128)
        
        # Residual Tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(128) for _ in range(num_res_blocks)
        ])
        
        self.flatten_dim = 128 * 8 * 8

        # --- CẬP NHẬT MỚI: Weight Initialization ---
        # Giúp model hội tụ nhanh hơn ngay từ đầu
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming He Initialization cho Conv layers (Tốt cho ReLU)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)
        x = x.view(-1, self.flatten_dim)
        return x

class PolicyNetwork(ChessResNet):
    """ACTOR: Output xác suất nước đi (4096)"""
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(self.flatten_dim, 1024)
        self.fc2 = nn.Linear(1024, 4096) 
        
        # Init lại layer cuối để tránh giá trị quá lớn ban đầu
        self._init_weights()

    def forward(self, x):
        features = self.forward_features(x)
        x = F.relu(self.fc1(features))
        logits = self.fc2(x)
        return logits

class ValueNetwork(ChessResNet):
    """CRITIC: Output giá trị thế cờ V(s) (1 scalar)"""
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc_value = nn.Linear(256, 1)
        
        self._init_weights()

    def forward(self, x):
        features = self.forward_features(x)
        x = F.relu(self.fc1(features))
        
        # Ép giá trị về [-1, 1] cho ổn định PPO Critic Loss
        value = torch.tanh(self.fc_value(x))
        return value