import linecache
import chess
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from time import perf_counter
import sqlite3

def _count_generator(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)

class LichessDataset(Dataset):
    """Lichess eval dataset."""

    cutoff = 800 # We dont really care about inaccuracies greater than this

    def _bound(self, val: float, min: float, max: float) -> float:
        if val < min:
            return min
        elif val > max:
            return max
        else:
            return val

    def __len__(self):
        return self._len

    def _get_encoding_eval(self, idx) -> tuple:
        raise NotImplementedError
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx > self.__len__():
            ValueError("Index exceedes limit")
        
        indicies, eval_raw = self._get_encoding_eval(idx)

        evl = self._bound(float(eval_raw), -self.cutoff, self.cutoff)

        sqp = np.zeros(64*2*6)
        for index_str in indicies:
            index = int(index_str)
            sqp[index] = 1
    
        sample = {'sqp': sqp, 'eval': evl}

        return sample

class LichessDatasetCSV(LichessDataset):
    """Lichess eval dataset."""

    def __init__(self, csv_file):
        """
        Arguments:
            csv_file (string): Path to the csv file with fen strings and evaluations.
        """

        self.csv_file = csv_file
        count = 0
        with open(self.csv_file, 'rb') as fp:
            c_generator = _count_generator(fp.raw.read)
            # count each \n
            count = sum(buffer.count(b'\n') for buffer in c_generator)
            fp.close()
        self._len = count
    
    def _get_encoding_eval(self, idx) -> tuple:
        entry = linecache.getline(self.csv_file, idx+1).split(',')
        indicies = entry[0]
        indicies = indicies.split(" ") # Split index list by spaces
        eval_raw = entry[5]

        return (indicies, eval_raw)
    
class LichessDatasetSQL(LichessDataset):
    """SQLITE lichess dataloader"""

    def __init__(self, db_file):
        """
        Arguments:
            db (string): Path to the sqlite db file with fen strings and evaluations.
        """

        self.db_file = db_file
        self.db_con = sqlite3.connect(db_file)
        self.cur = self.db_con.cursor()
        self._len = self.cur.execute("SELECT COUNT(*) FROM games;").fetchone()[0]
    
    def _get_encoding_eval(self, idx) -> tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx > self.__len__():
            ValueError("Index exceedes limit")
        entry = self.cur.execute('SELECT * FROM games WHERE [index] = ' + str(idx)).fetchone()

        indicies = entry[1]
        evl = self._bound(float(entry[2]), -self.cutoff, self.cutoff)


        indicies = indicies.split(" ") # Split index list by spaces
        return (indicies, evl)