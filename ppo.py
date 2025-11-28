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

from chess_utils import PolicyNetwork, ValueNetwork, board_to_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
KL_COEF = 0.05
EPSILON = 0.2
LR_ACTOR = 1e-5
LR_CRITIC = 1e-4
BATCH_SIZE = 16 

# Dataset lấy Prompt (Thế cờ khởi đầu)
class ChessPromptDataset(IterableDataset):
    def __init__(self, hf_dataset, max_samples=20000, min_elo=2000):
        self.dataset = hf_dataset
        self.max_samples = max_samples
        self.min_elo = min_elo

    def __iter__(self):
        count = 0
        print(f"-> PPO Prompt Iterator: Max {self.max_samples}, Elo > {self.min_elo}")
        for item in self.dataset:
            if count >= self.max_samples: break
            try:
                # 1. Lấy & Lọc & Clean
                content = item.get('pgn') or item.get('movetext')
                if not content: continue

                try:
                    w = int(re.search(r'\[WhiteElo "(\d+)"\]', content).group(1))
                    b = int(re.search(r'\[BlackElo "(\d+)"\]', content).group(1))
                    if w < self.min_elo or b < self.min_elo: continue
                except: pass

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
                
                # Lấy 1 thế cờ bất kỳ
                move_idx = random.randint(10, len(moves) - 1)
                board = chess.Board()
                valid = True
                for k in range(move_idx):
                    try: board.push(moves[k])
                    except: valid = False; break
                
                if not valid: continue
                count += 1
                yield board.fen()
            except: continue

def train_ppo_standard():
    print("--- BƯỚC 3: PPO RLHF (STANDARD GAMES) ---")
    
    # 1. Load Models (ResNet)
    actor = PolicyNetwork().to(DEVICE)
    # --- PHẦN QUAN TRỌNG: KIỂM TRA FILE CŨ ---
    if os.path.exists("ppo_chess_bot.pth"):
        print(">>> TÌM THẤY MODEL PPO CŨ. ĐANG LOAD ĐỂ HỌC TIẾP...")
        try:
            # Load trọng số đã học lần trước vào
            actor.load_state_dict(torch.load("ppo_chess_bot.pth", map_location=DEVICE))
            print(">>> Load thành công! Bot sẽ tiếp tục thông minh hơn.")
        except:
            # Phòng trường hợp file lỗi, quay về SFT gốc
            print("!!! File PPO lỗi. Load SFT gốc.")
            actor.load_state_dict(torch.load("sft_policy.pth", map_location=DEVICE))
            
    elif os.path.exists("sft_policy.pth"):
        # Nếu chưa từng chạy PPO, load kiến thức nền tảng SFT
        print(">>> Chưa có file PPO. Bắt đầu mới từ SFT.")
        actor.load_state_dict(torch.load("sft_policy.pth", map_location=DEVICE))
    # ------------------------------------------

    try: actor.load_state_dict(torch.load("sft_policy.pth", map_location=DEVICE))
    except: print("Thiếu sft_policy.pth"); return
    actor.train()
    
    ref_model = PolicyNetwork().to(DEVICE)
    ref_model.load_state_dict(torch.load("sft_policy.pth", map_location=DEVICE))
    ref_model.eval()
    
    critic = ValueNetwork().to(DEVICE)
    critic.train()
    
    reward_model = ValueNetwork().to(DEVICE)
    try: reward_model.load_state_dict(torch.load("reward_model.pth", map_location=DEVICE))
    except: print("Thiếu reward_model.pth"); return
    reward_model.eval()
    
    opt_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    opt_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)
    
    # 2. Dataset
    ds_raw = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    prompt_ds = ChessPromptDataset(ds_raw, max_samples=50000, min_elo=2000)
    dataloader = DataLoader(prompt_ds, batch_size=1) 
    
    states, actions, logprobs, rewards = [], [], [], []
    step_count = 0
    
    print("-> Bắt đầu PPO Loop...")
    for i, fen_batch in enumerate(dataloader):
        fen = fen_batch[0]
        board = chess.Board(fen)
        
        # --- ROLLOUT ---
        state_tensor = torch.from_numpy(board_to_tensor(board)).float().unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = actor(state_tensor)
            probs = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            ref_logits = ref_model(state_tensor)
            ref_probs = F.softmax(ref_logits, dim=1)
            ref_log_prob = torch.log(ref_probs[0][action] + 1e-10)
            kl_div = log_prob - ref_log_prob
            
        move_idx = action.item()
        final_reward = 0
        try:
            move = chess.Move(move_idx // 64, move_idx % 64)
            if move in board.legal_moves:
                board.push(move)
                next_state = torch.from_numpy(board_to_tensor(board)).float().unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    # CŨ: rm_score = reward_model(next_state).item()
                    raw_score = reward_model(next_state)
                    rm_score = torch.tanh(raw_score).item()
                # Reward tổng hợp
                final_reward = rm_score - (KL_COEF * kl_div.item())
            else:
                final_reward = -1.0 # Phạt nhẹ vì phạm luật (đã có model SFT cover)
        except:
            final_reward = -1.0

        states.append(state_tensor)
        actions.append(action)
        logprobs.append(log_prob)
        rewards.append(final_reward)
        
        # --- UPDATE ---
        if len(states) >= BATCH_SIZE:
            b_states = torch.cat(states)
            b_actions = torch.stack(actions)
            b_old_logprobs = torch.stack(logprobs).detach()
            b_rewards = torch.tensor(rewards).to(DEVICE).unsqueeze(1)
            
            values = critic(b_states)
            advantages = b_rewards - values.detach()
            
            # Actor Update
            curr_logits = actor(b_states)
            curr_probs = F.softmax(curr_logits, dim=1)
            curr_dist = torch.distributions.Categorical(curr_probs)
            curr_logprobs = curr_dist.log_prob(b_actions)
            
            ratio = torch.exp(curr_logprobs - b_old_logprobs.squeeze())
            surr1 = ratio * advantages.squeeze()
            surr2 = torch.clamp(ratio, 1-EPSILON, 1+EPSILON) * advantages.squeeze()
            actor_loss = -torch.min(surr1, surr2).mean()
            
            opt_actor.zero_grad()
            actor_loss.backward()
            opt_actor.step()
            
            # Critic Update
            new_values = critic(b_states)
            # critic_loss = F.mse_loss(new_values, b_rewards)
            critic_loss = F.smooth_l1_loss(new_values, b_rewards)
            
            opt_critic.zero_grad()
            critic_loss.backward()
            opt_critic.step()
            
            states, actions, logprobs, rewards = [], [], [], []
            step_count += 1
            
            if step_count % 10 == 0:
                print(f"Update {step_count} | A_Loss: {actor_loss.item():.4f} | C_Loss: {critic_loss.item():.4f}")

    torch.save(actor.state_dict(), "ppo_chess_bot.pth")
    print(">>> Đã cập nhật file: ppo_chess_bot.pth")

if __name__ == "__main__":
    train_ppo_standard()