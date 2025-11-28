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
import os

from chess_utils import ValueNetwork, board_to_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

# ... (Giữ nguyên class ChessStreamingPairwiseDataset như cũ) ...
# (Bạn copy lại class Dataset từ câu trả lời trước nhé, không cần sửa gì ở Dataset)
class ChessStreamingPairwiseDataset(IterableDataset):
    def __init__(self, hf_dataset, max_samples=50000, min_elo=2000):
        self.dataset = hf_dataset
        self.max_samples = max_samples
        self.min_elo = min_elo

    def __iter__(self):
        count = 0
        print(f"-> RM Iterator: Max {self.max_samples}, Elo > {self.min_elo}")
        for i, item in enumerate(self.dataset):
            if count >= self.max_samples: break
            try:
                content = item.get('pgn') or item.get('movetext')
                if not content: continue
                
                # Regex Clean
                content = re.sub(r'\{[^}]*\}', '', content)
                content = re.sub(r'\([^)]*\)', '', content)
                content = re.sub(r'\$\d+', '', content)

                moves = []
                if isinstance(content, str):
                    try:
                        pgn_io = io.StringIO(content)
                        game = chess.pgn.read_game(pgn_io)
                        if game and not game.errors: moves = list(game.mainline_moves())
                    except: continue

                if len(moves) < 15: continue
                move_idx = random.randint(10, len(moves) - 1)
                board = chess.Board()
                valid = True
                for k in range(move_idx):
                    try: board.push(moves[k])
                    except: valid=False; break
                if not valid: continue
                
                b_good = board.copy()
                try: b_good.push(moves[move_idx])
                except: continue
                x_good = torch.from_numpy(board_to_tensor(b_good)).float()

                b_bad = board.copy()
                legal = list(b_bad.legal_moves)
                if moves[move_idx] in legal: legal.remove(moves[move_idx])
                if not legal: continue
                b_bad.push(random.choice(legal))
                x_bad = torch.from_numpy(board_to_tensor(b_bad)).float()

                count += 1
                yield x_good, x_bad
            except: continue

def train_rm_fix():
    print("--- BƯỚC 2: REWARD MODEL (FIX COLLAPSE) ---")
    
    try:
        dataset = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    except: return

    train_ds = ChessStreamingPairwiseDataset(dataset, max_samples=100000, min_elo=2000)
    dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE)

    model = ValueNetwork().to(DEVICE)

    # --- THÊM ĐOẠN NÀY ---
    if os.path.exists("reward_model.pth"):
        print(">>> TÌM THẤY RM CŨ. ĐANG TẢI ĐỂ HỌC TIẾP...")
        try:
            model.load_state_dict(torch.load("reward_model.pth", map_location=DEVICE))
            print(">>> Đã tải thành công RM!")
        except:
            print("!!! Lỗi tải RM cũ. Train lại.")
    # ---------------------
    
    # --- THAY ĐỔI 1: GIẢM LEARNING RATE ---
    optimizer = optim.Adam(model.parameters(), lr=1e-5) # Giảm từ 1e-4 xuống 1e-5
    
    model.train()
    
    total_loss = 0
    batch_count = 0
    
    print("-> Bắt đầu train...")
    try:
        for i, (x_good, x_bad) in enumerate(dataloader):
            x_good, x_bad = x_good.to(DEVICE), x_bad.to(DEVICE)
            
            optimizer.zero_grad()
            
            r_good = model(x_good)
            r_bad = model(x_bad)
            
            loss = -torch.log(torch.sigmoid(r_good - r_bad)).mean()
            
            loss.backward()
            
            # --- THAY ĐỔI 2: GRADIENT CLIPPING (CHỐNG SỐC) ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if i % 50 == 0:
                # In thêm giá trị trung bình của Reward để debug
                mean_r = r_good.mean().item()
                diff = (r_good - r_bad).mean().item()
                print(f"Batch {i} | Loss: {loss.item():.4f} | R_mean: {mean_r:.3f} | Diff: {diff:.3f}")
                
                # Nếu Diff = 0.000 liên tục nghĩa là model đã chết
                if i > 100 and abs(diff) < 0.001:
                    print("!!! CẢNH BÁO: Model có dấu hiệu bị chết (Diff ~ 0).")

    except KeyboardInterrupt: print("stop")
    except Exception as e: print(f"Error: {e}")

    if batch_count > 0:
        print(f"=== KẾT THÚC | Avg Loss: {total_loss/batch_count:.4f} ===")
        torch.save(model.state_dict(), "reward_model.pth")
        print(">>> Đã lưu model (Fix version).")

if __name__ == "__main__":
    train_rm_fix()