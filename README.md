# ShallowLearn
Training repository for the ShallowRed NNUE using PyTorch
![Model training results](/resources/training.png)

The ShallowRed NNUE is trained on millions of board positions from the [lichess database](https://database.lichess.org/). This repo provides tools for the following.

1. Parsing PGN game data from lichess into a CSV format containing the board position and stockfish evaluation
2. Loading the CSV data into a sqlite database
3. PyTorch dataloaders for the CSV dataset and the sqlite database
4. Supervised training of models to estimate stockfish evaluations from board data

*Note:* training off the CSV is only recomended for small datasets, anything large and it'll be very slow to fetch data. Also keep in mind that the data storage is optimized for speed and not storage size, it is substantially larger than the PGN notation.

## Parsing raw lichess pgn data
Set the LICHESS_DATA environment variable to tell the script where the pgn data is located.

On linux: 

``` bash
export LICHESS_DATA="/home/username/Downloads/lichess_db_standard_rated_2023-08.pgn"
```

Before running the parsing script you'll need to build shallowred_interface. This provides python access to the evaluation function of the [ShallowRed chess engine](https://github.com/15jgme/shallow_red_engine). I plan to use this at somepoint to learn the difference between the stockfish eval and the traditional node evaluation used by ShallowRed
``` bash
# Make sure you're in the venv
cd sr_interface 
maturin dev # You'll probably need rust and cargo installed
cd ..
```

Then you can simply run the parsing script, to parse the pgn games into a CSV in your current directory

``` bash
python -m dataset.parse_lichess_sf.py
```

The script can take a while so you might want to run it in the background and send the results to a file

``` bash
nohup python -m dataset.parse_lichess_sf.py > parse_output.txt 2>&1 &
```

## Setting up the sqlite db
There is a dataloader which uses the plain csv file, but for anything large I suggest you create a sqlite db and load the csv into it.

``` bash
rm board_eval.db # Remove the existing db 
touch board_eval.db # Create the new db
python -m dataset.load_sqlite # Create a games table and load everything into it
```

## Training the model
With everything setup you can now train the model. I reccomend running it in the background and sending the results to a training_output.txt file.
``` bash
python train-model.py # Run in your terminal session
nohup python train-model.py > training_output.txt 2>&1 & # Run in the background (reccomended)
```
After the first iteration you'll see a model.pt checkpoint file being created. If it already exists, the train script will try to pickup from the checkpoint. The model will only create a new checkpoint if the current iteration is better on the test data.

## Development
These tools are pretty rough right now but will be improved gradually. Any PRs are very welcome. glhf, happy chess training