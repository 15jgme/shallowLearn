#!/home/jgme/Documents/software-projects/shallow-learn/.env/bin/python

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import numpy as np
from dataset.dataset import LichessDataset, LichessDatasetSQL
import matplotlib.pyplot as plt
import matplotlib

# torch.backends.cudnn.benchmark = True
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

# dataset = LichessDataset('games-sf-only.csv')
dataset = LichessDatasetSQL('games-sf.db')

batch_size =  64 #6384 #24 #256 #64 #32
validation_split = 0.2
# Creating data indices for training and validation splits:
dataset_size = len(dataset)
# dataset_size = 1000
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

def generate_sampler(split: int):
    shuffle_dataset = False
    random_seed = 42

    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    return(train_sampler, valid_sampler)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(64*2*6, 512)
        self.layer2 = nn.Linear(512, 32)
        self.layer3 = nn.Linear(32, 32)
        self.layer4 = nn.Linear(32, 1)

        # self.bnorm1 = nn.BatchNorm1d(512, affine=True)
        # self.bnorm2 = nn.BatchNorm1d(32, affine=True)
        # self.bnorm3 = nn.BatchNorm1d(32, affine=True)

    def forward(self, x):
        x = self.layer1(x)
        # x = self.bnorm1(x)
        x =  F.relu6(x) # Clipped relu

        x = self.layer2(x)
        # x = self.bnorm2(x)
        x = F.relu6(x)

        x = self.layer3(x)
        # x = self.bnorm3(x)
        x = F.relu6(x)

        x = self.layer4(x)

        return x
    
test_losses = np.array([])
epoch_batch_losses = np.array([])

# Plotting
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

def plot_losses(avg_loss_epochs, loss_data):
    fig = plt.figure()
    # fig, axis = plt.subplots(2)
    # plt.xlabel('Epoch')
    # plt.ylabel('Test Loss')
    # fig.clear()
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(avg_loss_epochs)
    ax1.set_title("Losses over epochs")

    ax2 = fig.add_subplot(2,1,2)
    counts, bins = np.histogram(loss_data)
    ax2.stairs(counts, bins)
    ax2.set_title("Last epoch loss histogram")

    fig.savefig("training.png")
    # Take 100 episode averages and plot them too

    # plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.display(fig.gcf())
        display.clear_output(wait=True)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, sample in enumerate(dataloader):
        optimizer.zero_grad()
        x = sample['sqp'].to(device).type(torch.float64) # should be to sparse
        y = sample['eval'].to(device)

        predic = model(x).squeeze()
        loss = loss_fn(predic, y)

        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    global test_losses
    global epoch_batch_losses
    model.eval()
    epoch_batch_losses = np.array([]) # Clear epoch histo array
    num_batches = len(dataloader)
    test_loss = 0
    
    with torch.no_grad():
        for sample in dataloader:
            x = sample['sqp'].to(device).type(torch.float64) # should be to sparse
            y = sample['eval'].to(device)

            predic = model(x).squeeze()
            current_loss = loss_fn(predic, y).item()
            test_loss += current_loss
            epoch_batch_losses = np.append(epoch_batch_losses, current_loss)

        test_loss /= num_batches
        test_losses = np.append(test_losses, test_loss)
        print(f"Avg loss: {test_loss:>8f} \n")

model = NeuralNetwork().to(device)
# Initialize model, loss, and optimizer
min_loss = 10000000000000000 # Really big number
try:
    checkpoint = torch.load('model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    try:
        min_loss = checkpoint['loss']
    except Exception as e:
        print(e)
    try:
        split = checkpoint['split']
    except Exception as e:
        print(e)
except Exception as e:
    print(e)

train_sampler, valid_sampler = generate_sampler(split)
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)
validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=2, pin_memory=True)

loss_fn = nn.L1Loss()
# loss_fn = nn.SmoothL1Loss()
# loss_fn = nn.MSELoss()
learning_rate = 1e-4
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, amsgrad=True)
# optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = ReduceLROnPlateau(optimizer, 'min')
# scheduler = CyclicLR(optimizer, base_lr=learning_rate, max_lr=0.01)

epochs = 10000

test_loop(validation_loader, model, loss_fn)
plot_losses(test_losses, epoch_batch_losses)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(validation_loader, model, loss_fn)
    # scheduler.step(test_losses[-1])
    # scheduler.step()
    plot_losses(test_losses, epoch_batch_losses)

    if(test_losses[-1] < min_loss):
        # Save the model only if it's the best one
        torch.save({
            'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': min_loss,
            'split': split, # Save the model split so that we do not test on training data after a training restart
            }, 'model.pt')
    
print("Done!")