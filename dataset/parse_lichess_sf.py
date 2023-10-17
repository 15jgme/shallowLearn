import chess.pgn
# from shallowred_interface import fen_eval
import codecs
from tqdm import tqdm
import numpy as np
from typing import List
from .parse_util import board_to_int
import os
import sr_interface

data_file = os.environ['LICHESS_DATA']
output_file = "games-sf-only.csv"

continue_from_index = 0
game_count_limit = 500000

if continue_from_index == 0:
    # Only overwrite if beginning at zero
    with codecs.open(output_file, 'w', 'utf-8') as f:
        f.close()

pgn = open(data_file)

def generate_csv_lines(current_game: chess.pgn.Game) -> (List[str] | None):
    fen_eval_lines = []
    board = current_game.board()
    move_count = 0 # Number of moves placed in game
    opening_count = 0 # Six moves expected during the opening 
    for node in current_game.mainline():
        eval = node.eval()
        board = node.board()
        if eval is None:
            pass # We don't have an eval for this move
        else:
            is_mate = 0
            if eval.is_mate():
                # If this move is a forced mate
                is_mate = 1
                cp_w_score_mate = eval.white().mate()
                baseline_component = 4000 if cp_w_score_mate > 0 else -4000
                mate_distance_component = 0
                if abs(cp_w_score_mate) < 10:
                    mate_distance = (10-cp_w_score_mate) if cp_w_score_mate > 0 else (-10+cp_w_score_mate)
                    mate_distance_component = 50*mate_distance
                cp_w_score = baseline_component + mate_distance_component
            else:
                # If normal move (non-forced)
                cp_w_score = eval.white().score() # Get the centipawn score for white

            fen = board.fen()
            sqp = board_to_int(board)
            shallowred_w_score = sr_interface.fen_eval_material(fen)
            heur_w_score = cp_w_score - shallowred_w_score

            # Should present eval for the current player
            # Valid since if black: heur_b_score = -heur_w_score = -(cp_w_score - shallowred_w_score)
            heur_side_score = heur_w_score if board.turn == chess.WHITE else -heur_w_score
            cp_side_score = cp_w_score if board.turn == chess.WHITE else -cp_w_score
            
            save_move = not node.board().is_checkmate()
            try:
                if save_move:
                    fen_eval_lines.append("%s,%d,%d,%d,%d,%d,%r\n"%(sqp, heur_side_score, cp_w_score, shallowred_w_score, is_mate, cp_side_score, board.turn))
            except Exception as e:
                print(e)
                print(eval)
            move_count += 1 # Increment count
        
    if fen_eval_lines == []:
        return None
    else:
        return fen_eval_lines

game_count = 0

for i in tqdm(range(game_count_limit), ascii=True, desc="Searching for games with eval"):
    if i < continue_from_index:
        continue
    
    while True:
        current_game = chess.pgn.read_game(pgn)
        if current_game is None:
            # print("Parsing complete, all games parsed")
            break
        else:
            # print("Looking at %s" % (current_game.headers.get('Event')))
            eval_lines = generate_csv_lines(current_game)
            # Write line to file
            if eval_lines is not None:
                with codecs.open(output_file, 'a', 'utf-8') as f:
                    for line in eval_lines:
                        f.write(line)
                    f.close()
                break   

        




