import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import chess
import chess.pgn
import io
import random
import re
import os
import logging
logging.getLogger("chess").setLevel(logging.CRITICAL)
from chess_utils import PolicyNetwork, ValueNetwork, board_to_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- HYPERPARAMETERS ĐÃ TINH CHỈNH ---
KL_COEF = 0.02       # Giảm phạt KL để Bot tự do hơn
ENTROPY_COEF = 0.01  # Tăng tò mò (Rất quan trọng)
EPSILON = 0.2
LR_ACTOR = 1e-5      # Học chậm mà chắc
LR_CRITIC = 1e-4
BATCH_SIZE = 16

def clean_pgn(content):
    content = re.sub(r'\{[^}]*\}', ' ', content)
    content = re.sub(r'\$\d+', ' ', content)
    while '(' in content: content = re.sub(r'\([^()]*\)', ' ', content)
    return " ".join(content.split())

def get_legal_mask(board):
    mask = torch.full((1, 4096), -float('inf'))
    indices = [m.from_square*64 + m.to_square for m in board.legal_moves]
    # Filter indices >= 4096 (phòng hờ)
    indices = [i for i in indices if i < 4096]
    if indices: mask[0, indices] = 0
    return mask.to(DEVICE), bool(indices)

class ChessPPODataset(IterableDataset):
    def __init__(self, hf_dataset, max_samples=50000, min_elo=2000):
        self.dataset = hf_dataset
        self.max_samples = max_samples
        self.min_elo = min_elo
    def __iter__(self):
        count = 0
        print(f"-> PPO Start: Max {self.max_samples}")
        for item in self.dataset:
            if count >= self.max_samples: break
            try:
                we = item.get('WhiteElo'); be = item.get('BlackElo')
                if we and be and (int(we) < self.min_elo or int(be) < self.min_elo): continue
                content = item.get('pgn') or item.get('movetext')
                if not content: continue
                
                moves = []
                pgn_io = io.StringIO(clean_pgn(content))
                game = chess.pgn.read_game(pgn_io)
                if game and not game.errors: moves = list(game.mainline_moves())
                
                if len(moves) < 15: continue
                move_idx = random.randint(10, len(moves)-1)
                board = chess.Board()
                valid = True
                for k in range(move_idx):
                    try: board.push(moves[k])
                    except: valid=False; break
                if not valid: continue
                
                count += 1
                yield board.fen()
            except: continue

def train_ppo():
    print("--- BƯỚC 3: PPO (ENTROPY BOOST) ---")
    
    actor = PolicyNetwork().to(DEVICE)
    # Load Resume
    if os.path.exists("ppo_chess_bot.pth"):
        try: actor.load_state_dict(torch.load("ppo_chess_bot.pth", map_location=DEVICE)); print(">>> Resumed PPO.")
        except: actor.load_state_dict(torch.load("sft_policy.pth", map_location=DEVICE))
    elif os.path.exists("sft_policy.pth"):
        actor.load_state_dict(torch.load("sft_policy.pth", map_location=DEVICE))
    else: return
    actor.train()
    
    ref_model = PolicyNetwork().to(DEVICE)
    try: ref_model.load_state_dict(torch.load("sft_policy.pth", map_location=DEVICE))
    except: return
    ref_model.eval()
    
    critic = ValueNetwork().to(DEVICE)
    critic.train()
    
    rm = ValueNetwork().to(DEVICE)
    try: rm.load_state_dict(torch.load("reward_model.pth", map_location=DEVICE))
    except: return
    rm.eval()
    
    opt_act = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    opt_cri = optim.Adam(critic.parameters(), lr=LR_CRITIC)
    
    try: ds = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    except: return
    dl = DataLoader(ChessPPODataset(ds), batch_size=1)
    
    states, actions, logprobs, rewards = [], [], [], []
    step = 0
    tot_rew = 0
    
    print("-> Training Loop...")
    for i, fen in enumerate(dl):
        board = chess.Board(fen[0])
        state = torch.from_numpy(board_to_tensor(board)).float().unsqueeze(0).to(DEVICE)
        mask, valid = get_legal_mask(board)
        if not valid: continue
        
        # ROLLOUT
        with torch.no_grad():
            logits = actor(state) + mask
            probs = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            lp = dist.log_prob(action)
            
            # KL
            ref_logits = ref_model(state) + mask
            ref_probs = F.softmax(ref_logits, dim=1)
            ref_lp = torch.log(ref_probs.gather(1, action.unsqueeze(1)).squeeze(1) + 1e-10)
            kl = lp - ref_lp
            
        # ACT
        idx = action.item()
        try:
            move = chess.Move(idx//64, idx%64)
            if move in board.legal_moves:
                board.push(move)
                ns = torch.from_numpy(board_to_tensor(board)).float().unsqueeze(0).to(DEVICE)
                with torch.no_grad(): 
                    raw_score = rm(ns)
                    score = torch.tanh(raw_score).item() # Ép về [-1, 1]
                
                # Reward = RM - KL (Chặn KL âm)
                # Max(0, KL) để không thưởng cho việc đi lùi
                rew = score - KL_COEF * max(0.0, kl.item())
            else: rew = -2.0
        except: rew = -2.0
        
        states.append(state); actions.append(action); logprobs.append(lp); rewards.append(rew)
        tot_rew += rew
        
        # UPDATE
        if len(states) >= BATCH_SIZE:
            bs = torch.cat(states)
            ba = torch.stack(actions).squeeze(-1) # [16]
            blp = torch.stack(logprobs).squeeze(-1).detach() # [16]
            br = torch.tensor(rewards).to(DEVICE).unsqueeze(1) # [16, 1]
            
            # Advantage
            vals = critic(bs)
            adv = br - vals.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            
            # Actor Loss (with Entropy)
            c_logits = actor(bs) 
            # Note: Update loop ko dùng mask để tiết kiệm, giả định log_prob focus đúng index
            c_probs = F.softmax(c_logits, dim=1)
            c_dist = torch.distributions.Categorical(c_probs)
            c_lp = c_dist.log_prob(ba)
            
            ratio = torch.exp(c_lp - blp)
            surr1 = ratio * adv.squeeze()
            surr2 = torch.clamp(ratio, 1-EPSILON, 1+EPSILON) * adv.squeeze()
            
            entropy = c_dist.entropy().mean()
            # Loss = PPO - Entropy_Bonus (Dấu trừ để Maximize Entropy)
            a_loss = -torch.min(surr1, surr2).mean() - (ENTROPY_COEF * entropy)
            
            opt_act.zero_grad()
            a_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            opt_act.step()
            
            # Critic Loss
            new_vals = critic(bs)
            c_loss = F.smooth_l1_loss(new_vals, br)
            opt_cri.zero_grad()
            c_loss.backward()
            opt_cri.step()
            
            states, actions, logprobs, rewards = [], [], [], []
            step += 1
            
            if step % 10 == 0:
                print(f"Step {step} | Rew: {tot_rew/(10*BATCH_SIZE):.3f} | Ent: {entropy.item():.3f} | RM_Last: {score:.3f}")
                tot_rew = 0
            
            if step >= 50000: break
            
    torch.save(actor.state_dict(), "ppo_chess_bot.pth")
    print(">>> Saved PPO.")

if __name__ == "__main__":
    train_ppo()