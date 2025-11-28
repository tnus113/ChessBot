import flask
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
import chess
import os
import random

from chess_utils import PolicyNetwork, board_to_tensor, encode_move

app = Flask(__name__, template_folder='templates')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = PolicyNetwork().to(DEVICE)
MODEL_PATH = None

if os.path.exists("ppo_chess_bot.pth"): MODEL_PATH = "ppo_chess_bot.pth"
elif os.path.exists("sft_policy.pth"): MODEL_PATH = "sft_policy.pth"

if MODEL_PATH:
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f">>> SERVER READY: {MODEL_PATH}")
    except Exception as e:
        print(f"!!! LOAD ERROR: {e}")
        MODEL_PATH = None
else:
    print("!!! WARNING: No model found.")

@app.route('/')
def index(): return render_template('index.html')

@app.route('/get_move', methods=['POST'])
def get_move():
    try:
        data = request.json
        fen = data.get('fen')
        board = chess.Board(fen)
        
        if board.is_game_over():
            return jsonify({'move': None, 'game_over': True})

        legal_moves = list(board.legal_moves)
        best_move = None
        
        if MODEL_PATH:
            tensor = torch.from_numpy(board_to_tensor(board)).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(tensor)
            
            best_score = -float('inf')
            for move in legal_moves:
                idx = encode_move(move)
                if idx < 4096:
                    score = logits[0][idx].item()
                    if score > best_score:
                        best_score = score
                        best_move = move
            
            if not best_move: best_move = random.choice(legal_moves)
            print(f"Bot: {best_move.uci()} ({best_score:.2f})")
        else:
            best_move = random.choice(legal_moves)

        return jsonify({'move': best_move.uci(), 'game_over': False})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)