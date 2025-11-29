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
from chess_utils import PolicyNetwork, board_to_tensor, encode_move

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

def clean_pgn(content):
    content = re.sub(r'\{[^}]*\}', ' ', content)
    content = re.sub(r'\$\d+', ' ', content)
    while '(' in content:
        content = re.sub(r'\([^()]*\)', ' ', content)
    return " ".join(content.split())

class ChessSFTDataset(IterableDataset):
    def __init__(self, hf_dataset, max_samples=100000, min_elo=2000):
        self.dataset = hf_dataset
        self.max_samples = max_samples
        self.min_elo = min_elo

    def __iter__(self):
        count = 0
        print(f"-> SFT Start: Max {self.max_samples}")
        for item in self.dataset:
            if count >= self.max_samples: break
            try:
                # Filter Elo
                we = item.get('WhiteElo'); be = item.get('BlackElo')
                if we and be and (int(we) < self.min_elo or int(be) < self.min_elo): continue
                
                content = item.get('pgn') or item.get('movetext') or item.get('moves')
                if not content: continue
                
                moves = []
                pgn_io = io.StringIO(clean_pgn(content))
                game = chess.pgn.read_game(pgn_io)
                if game and not game.errors: moves = list(game.mainline_moves())
                
                if len(moves) < 15: continue
                move_idx = random.randint(10, len(moves) - 1)
                
                board = chess.Board()
                for k in range(move_idx): board.push(moves[k])
                
                x = torch.from_numpy(board_to_tensor(board)).float()
                y = encode_move(moves[move_idx])
                count += 1
                yield x, y
            except: continue

def train_sft():
    print("--- BƯỚC 1: SFT ---")
    try: dataset = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    except: return
    
    train_ds = ChessSFTDataset(dataset, max_samples=100000)
    dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE)
    
    model = PolicyNetwork().to(DEVICE)
    # Load cũ nếu có
    import os
    if os.path.exists("sft_policy.pth"):
        try: model.load_state_dict(torch.load("sft_policy.pth", map_location=DEVICE)); print(">>> Loaded SFT.")
        except: pass

    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Giảm LR để học kỹ hơn
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    total_loss = 0
    batch_count = 0
    
    for i, (b, l) in enumerate(dataloader):
        b, l = b.to(DEVICE), l.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(b), l)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
        if i % 100 == 0: print(f"Batch {i} | Loss: {loss.item():.4f}")
        if i >= 3000: break # Train 3000 batch rồi nghỉ
        
    if batch_count > 0:
        print(f"Avg Loss: {total_loss/batch_count:.4f}")
        torch.save(model.state_dict(), "sft_policy.pth")
        print(">>> Saved SFT.")

if __name__ == "__main__":
    train_sft()