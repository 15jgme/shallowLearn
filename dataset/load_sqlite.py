import pandas as pd
import sqlite3

df = pd.read_csv('games-sf-only.csv', names=['bits', 'side_heuristic_cp', 'eval_w', 'sr_score_w', 'forced_mate', 'eval_side', 'white_to_move'])
print('dataframe loaded')
connection = sqlite3.connect('games-sf.db')
print(df.index)

df.to_sql('games', connection, if_exists='replace', index=True)

connection.close()