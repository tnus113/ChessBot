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

# Import utils từ file chess_utils.py (Đã có sẵn)
from chess_utils import PolicyNetwork, board_to_tensor, encode_move

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

class ChessStreamingDataset(IterableDataset):
    def __init__(self, hf_dataset, max_samples=100000, min_elo=2000):
        self.dataset = hf_dataset
        self.max_samples = max_samples
        self.min_elo = min_elo

    def __iter__(self):
        count = 0
        print(f"-> SFT Iterator: Mục tiêu {self.max_samples} mẫu, Elo > {self.min_elo}")
        
        for i, item in enumerate(self.dataset):
            if count >= self.max_samples: break

            try:
                # 1. LẤY NỘI DUNG PGN
                # Dataset này thường dùng key 'pgn'
                content = item.get('pgn') or item.get('movetext')
                if not content: continue

                # 2. LỌC ELO (Xử lý trực tiếp trên chuỗi PGN để nhanh hơn)
                # Tìm chuỗi [WhiteElo "xxxx"] và [BlackElo "xxxx"]
                try:
                    w_elo_match = re.search(r'\[WhiteElo "(\d+)"\]', content)
                    b_elo_match = re.search(r'\[BlackElo "(\d+)"\]', content)
                    
                    if w_elo_match and b_elo_match:
                        w_elo = int(w_elo_match.group(1))
                        b_elo = int(b_elo_match.group(1))
                        if w_elo < self.min_elo or b_elo < self.min_elo:
                            continue
                except: pass

                # 3. LÀM SẠCH PGN (QUAN TRỌNG)
                # Xóa comment { ... }, biến thể ( ... ), và các tag khác
                content_clean = re.sub(r'\{[^}]*\}', '', content)
                content_clean = re.sub(r'\([^)]*\)', '', content_clean)
                content_clean = re.sub(r'\$\d+', '', content_clean)
                
                # 4. PARSE GAME
                moves = []
                if isinstance(content_clean, str):
                    try:
                        pgn_io = io.StringIO(content_clean)
                        game = chess.pgn.read_game(pgn_io)
                        # Kiểm tra lỗi parse
                        if game and not game.errors:
                            moves = list(game.mainline_moves())
                    except: continue

                if len(moves) < 15: continue 

                # 5. CHỌN MẪU NGẪU NHIÊN (Mid-game)
                # Chọn từ nước 10 trở đi
                move_idx = random.randint(10, len(moves) - 1)
                
                board = chess.Board()
                valid_game = True
                
                # Replay ván cờ
                for k in range(move_idx):
                    try:
                        board.push(moves[k])
                    except:
                        valid_game = False
                        break
                
                if not valid_game: continue

                x = torch.from_numpy(board_to_tensor(board)).float()
                y = encode_move(moves[move_idx])
                
                count += 1
                yield x, y

            except Exception:
                continue

def train_sft_standard():
    print("--- BƯỚC 1: SFT (LICHESS STANDARD GAMES) ---")
    print(f"-> Thiết bị: {DEVICE}")
    
    print("-> Đang kết nối Streaming...")
    try:
        # dataset này rất lớn, bắt buộc dùng streaming=True
        dataset = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    except Exception as e:
        print(f"Lỗi kết nối: {e}")
        return

    # Train 100k mẫu
    MAX_SAMPLES = 100000
    train_ds = ChessStreamingDataset(dataset, max_samples=MAX_SAMPLES, min_elo=2000)
    dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE)
    
    model = PolicyNetwork().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    total_loss = 0
    batch_count = 0
    start_time = time.time()
    
    print("-> Bắt đầu train...")
    try:
        for i, (boards, labels) in enumerate(dataloader):
            boards, labels = boards.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(boards)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if i % 50 == 0:
                elapsed = time.time() - start_time
                speed = (i * BATCH_SIZE) / elapsed if elapsed > 0 else 0
                print(f"Batch {i} | Loss: {loss.item():.4f} | Speed: {speed:.1f} samples/s")
            
            if i >= (MAX_SAMPLES // BATCH_SIZE):
                print("-> Đạt mục tiêu mẫu. Dừng train.")
                break

    except KeyboardInterrupt:
        print("-> Dừng thủ công.")
    except Exception as e:
        print(f"-> Lỗi: {e}")

    if batch_count > 0:
        print(f"=== Avg Loss: {total_loss/batch_count:.4f} ===")
        torch.save(model.state_dict(), "sft_policy.pth")
        print(">>> Đã lưu: sft_policy.pth")

if __name__ == "__main__":
    train_sft_standard()