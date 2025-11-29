# ... (Giữ nguyên import và clean_pgn từ file trước) ...
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import chess
import chess.pgn
import io
import random
import re
import os
from chess_utils import ValueNetwork, board_to_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

def clean_pgn(content):
    content = re.sub(r'\{[^}]*\}', ' ', content)
    content = re.sub(r'\$\d+', ' ', content)
    while '(' in content: content = re.sub(r'\([^()]*\)', ' ', content)
    return " ".join(content.split())

class ChessRMPairDataset(IterableDataset):
    def __init__(self, hf_dataset, max_samples=50000):
        self.dataset = hf_dataset
        self.max_samples = max_samples

    def __iter__(self):
        count = 0
        for item in self.dataset:
            if count >= self.max_samples: break
            try:
                content = item.get('pgn') or item.get('movetext')
                if not content: continue
                moves = []
                pgn_io = io.StringIO(clean_pgn(content))
                game = chess.pgn.read_game(pgn_io)
                if game and not game.errors: moves = list(game.mainline_moves())
                
                if len(moves) < 20: continue
                move_idx = random.randint(10, len(moves) - 1)
                
                board = chess.Board()
                for k in range(move_idx): board.push(moves[k])
                
                # Good Action (GM Move)
                b_good = board.copy()
                correct_move = moves[move_idx]
                b_good.push(correct_move)
                x_good = torch.from_numpy(board_to_tensor(b_good)).float()
                
                # Bad Action (Random Legal Move)
                # Đây là cách duy nhất nếu không có Engine: Giả định GM luôn đúng
                b_bad = board.copy()
                legal = list(b_bad.legal_moves)
                if correct_move in legal: legal.remove(correct_move)
                if not legal: continue
                
                random_move = random.choice(legal)
                b_bad.push(random_move)
                x_bad = torch.from_numpy(board_to_tensor(b_bad)).float()
                
                count += 1
                yield x_good, x_bad
            except: continue

def train_rm():
    print("--- BƯỚC 2: REWARD MODEL ---")
    try: dataset = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    except: return
    
    dataloader = DataLoader(ChessRMPairDataset(dataset), batch_size=BATCH_SIZE)
    model = ValueNetwork().to(DEVICE)
    
    if os.path.exists("reward_model.pth"):
        try: model.load_state_dict(torch.load("reward_model.pth", map_location=DEVICE)); print(">>> Loaded RM.")
        except: pass
        
    # LR nhỏ để học kỹ
    optimizer = optim.Adam(model.parameters(), lr=5e-5) 
    model.train()
    
    for i, (g, b) in enumerate(dataloader):
        g, b = g.to(DEVICE), b.to(DEVICE)
        optimizer.zero_grad()
        rg = model(g); rb = model(b)
        
        # Loss + Regularization (Giữ điểm số không bị trôi)
        loss = -torch.log(torch.sigmoid(rg - rb)).mean() + 0.001 * (rg**2 + rb**2).mean()
        
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            diff = (rg - rb).mean().item()
            print(f"Batch {i} | Loss: {loss.item():.4f} | Diff: {diff:.3f}")
        if i >= 3000: break
        
    torch.save(model.state_dict(), "reward_model.pth")
    print(">>> Saved RM.")

if __name__ == "__main__":
    train_rm()