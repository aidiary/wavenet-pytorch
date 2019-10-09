import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from data import WaveNetDataset
from model import WaveNet

from tqdm import tqdm
from tensorboardX import SummaryWriter


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(54321)
torch.cuda.manual_seed(54321)

gpu_id = 2
device = torch.device('cuda:{}'.format(gpu_id)
                      if torch.cuda.is_available() else 'cpu')

batch_size = 8
lr = 1e-3
epochs = 100000
logging_iters = 10
checkpoint_iters = 1000
writer = SummaryWriter()


def main():
    train_dataset = WaveNetDataset('train.list')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=0,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=True)

    model = WaveNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # training
    model.train()
    global_iters = 0
    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch))
        for mel, audio in tqdm(train_loader, total=len(train_loader), desc='train'):
            mel = mel.to(device)
            audio = audio.to(device)
            optimizer.zero_grad()
            audio_pred = model(mel, audio)
            loss = criterion(audio_pred, audio)
            loss.backward()
            optimizer.step()

            if global_iters % logging_iters == 0:
                writer.add_scalar('train/loss', loss.item(), global_iters)

            if global_iters % checkpoint_iters == 0:
                checkpoint_path = os.path.join(
                    writer.logdir, 'checkpoint_{}.pth'.format(global_iters))
                save_checkpoint(model, optimizer,
                                global_iters, checkpoint_path)

            global_iters += 1


def save_checkpoint(model, optimizer, iteration, checkpoint_path):
    state_dict = model.state_dict()
    state = {
        'model': state_dict,
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(state, checkpoint_path)


if __name__ == "__main__":
    main()
