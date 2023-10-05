from .dataset import LichessDataset
from .parse_util import orient, board_to_int

import numpy as np
import chess
import unittest
from time import perf_counter

class TestStringMethods(unittest.TestCase):

    def test_orient(self):
        self.assertEqual(orient(chess.WHITE, chess.A1), chess.A1)
        self.assertEqual(orient(chess.BLACK, chess.A1), chess.H8)
        self.assertNotEqual(orient(chess.BLACK, chess.A1), orient(chess.WHITE, chess.A1))

    def test_mirror(self):
        mirror_fen_w = 'rnb2bnr/pppk1ppp/3p4/1q2p3/3P2Q1/4P3/PPP1KPPP/RNB2BNR w - - 0 1' # A position that looks the same for white and black, white to move
        mirror_fen_b = 'rnb2bnr/pppk1ppp/3p4/1q2p3/3P2Q1/4P3/PPP1KPPP/RNB2BNR b - - 0 1' # A position that looks the same for white and black, black to move

        mirror_w = board_to_int(chess.Board(fen=mirror_fen_w)).split(' ')
        mirror_b =  board_to_int(chess.Board(fen=mirror_fen_b)).split(' ')
        self.assertTrue(set(mirror_w) == set(mirror_b)) # Confirm that the list of non-zero indicies are the same (ignoring order)


if __name__ == '__main__':
    unittest.main()