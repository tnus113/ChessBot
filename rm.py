import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import chess
import chess.pgn
import io
import random
import sys
import time
import re

from chess_utils import ValueNetwork, board_to_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

class ChessStreamingPairwiseDataset(IterableDataset):
    def __init__(self, hf_dataset, max_samples=50000, min_elo=2000):
        self.dataset = hf_dataset
        self.max_samples = max_samples
        self.min_elo = min_elo

    def __iter__(self):
        count = 0
        print(f"-> RM Iterator: Mục tiêu {self.max_samples} cặp")
        
        for i, item in enumerate(self.dataset):
            if count >= self.max_samples: break

            try:
                # 1. Lấy & Lọc PGN
                content = item.get('pgn') or item.get('movetext')
                if not content: continue

                # Regex lọc Elo nhanh
                try:
                    w = int(re.search(r'\[WhiteElo "(\d+)"\]', content).group(1))
                    b = int(re.search(r'\[BlackElo "(\d+)"\]', content).group(1))
                    if w < self.min_elo or b < self.min_elo: continue
                except: pass

                # 2. Làm sạch & Parse
                content = re.sub(r'\{[^}]*\}', '', content)
                content = re.sub(r'\([^)]*\)', '', content)
                content = re.sub(r'\$\d+', '', content)

                moves = []
                if isinstance(content, str):
                    try:
                        pgn_io = io.StringIO(content)
                        game = chess.pgn.read_game(pgn_io)
                        if game and not game.errors:
                            moves = list(game.mainline_moves())
                    except: continue

                if len(moves) < 15: continue

                # 3. Tạo cặp Good/Bad
                move_idx = random.randint(10, len(moves) - 1)
                
                board = chess.Board()
                valid = True
                for k in range(move_idx):
                    try: board.push(moves[k])
                    except: 
                        valid = False; break
                if not valid: continue
                
                # Good (Thực tế)
                b_good = board.copy()
                correct_move = moves[move_idx]
                try: b_good.push(correct_move)
                except: continue
                x_good = torch.from_numpy(board_to_tensor(b_good)).float()

                # Bad (Random)
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

def train_rm_standard():
    print("--- BƯỚC 2: REWARD MODEL (STANDARD GAMES) ---")
    
    try:
        dataset = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    except Exception as e:
        print(f"Lỗi: {e}")
        return

    train_ds = ChessStreamingPairwiseDataset(dataset, max_samples=50000, min_elo=2000)
    dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE)

    model = ValueNetwork().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    
    total_loss = 0
    batch_count = 0
    
    print("-> Bắt đầu train RM...")
    try:
        for i, (x_good, x_bad) in enumerate(dataloader):
            x_good, x_bad = x_good.to(DEVICE), x_bad.to(DEVICE)
            
            optimizer.zero_grad()
            
            r_good = model(x_good) # [-1, 1]
            r_bad = model(x_bad)   # [-1, 1]
            
            # Loss ép điểm Good > Bad
            loss = -torch.log(torch.sigmoid(r_good - r_bad)).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if i % 50 == 0:
                print(f"Batch {i} | Loss: {loss.item():.4f}")
            
            if i >= (50000 // BATCH_SIZE): break

    except KeyboardInterrupt: print("-> Dừng thủ công.")
    except Exception as e: print(f"-> Lỗi: {e}")

    if batch_count > 0:
        print(f"=== Avg Loss: {total_loss/batch_count:.4f} ===")
        torch.save(model.state_dict(), "reward_model.pth")
        print(">>> Đã lưu: reward_model.pth")

if __name__ == "__main__":
    train_rm_standard()