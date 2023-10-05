import numpy as np
from typing import List
import chess

piece_map_own = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
piece_map_enemy = {chess.PAWN: 6, chess.KNIGHT: 7, chess.BISHOP: 8, chess.ROOK: 9, chess.QUEEN: 10, chess.KING: 11}

def orient(is_white_pov: bool, sq: int):
  return (63 * (not is_white_pov)) ^ sq

def board_to_int(board: chess.Board) -> str:
    """
    board: chess.Board -> Input board to be serialized
    return string -> indicies of nonzero bits sperated by spaces
    """
    sqp = ""
    for square, piece in board.piece_map().items():
        sq_reoriented = orient(board.turn, square)
        index = 0
        if piece.color == board.turn:
            # Friendly piece
            index = piece_map_own[piece.piece_type] * 64 + sq_reoriented
        else:
            # Enemy piece
            index = piece_map_enemy[piece.piece_type] * 64 + sq_reoriented
        sqp += str(index) + " "


    sqp = sqp[:-1] # Remove the last space
    return sqp