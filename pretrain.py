import argparse
import torch
from utils import get_loader, VGG16, test_l
import torch.nn as nn
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Pretrain magnitude task')

parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout')
parser.add_argument('--optimizer', default="adam", type=str, help="otimizer algorithm")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device", flush=True)
  
class CustomMAE(nn.Module):
    def __init__(self):
        super(CustomMAE, self).__init__()
        self.mae = mae = nn.L1Loss()

    def forward(self, inputs, targets):
        mvalue = 99/35
        mask = torch.tensor(targets != mvalue, dtype=torch.float32)
        return self.mae(inputs*mask, targets*mask)

def main():
    args = parser.parse_args()
    print(args, flush=True)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    #LOAD DATA
    train_loader, val_loader, test_loader = get_loader("unl", args.batch_size)

    #LOAD MODEL
    model = VGG16(12, args.dropout)

    #TRAIN
    model = model.to(device)
    criterion = CustomMAE()
    if args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    model.train()

    train_loss = []
    test_loss = []
    best_loss = 1e10

    for n_epoch in range(args.epochs):
        epoch_train_loss = 0
        num_batch_count = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            opt.zero_grad()
            out = model(images)
            print(labels.shape, flush=True)
            print(images.shape, flush=True)

            loss = criterion(out, labels)
            loss.backward()
            opt.step()

            epoch_train_loss += loss.item()
            num_batch_count +=1

        val_loss = test_l(test_loader,model, criterion, device)
        print(f"[{n_epoch+1}/{args.epochs}] - Training loss: {epoch_train_loss/num_batch_count} - Validation loss: {val_loss}", flush=True)
        train_loss.append(epoch_train_loss/num_batch_count)
        test_loss.append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': n_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': criterion,
                }, args.save_dir + "/checkpoint.pth")

    plt.figure(figsize=(8, 6))
    plt.plot(list(range(args.epochs)), test_loss, marker='o', linestyle='-', label='Validation Loss')
    plt.title('Validation and Training Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(list(range(args.epochs)), train_loss, marker='o', linestyle='-', label='Training Loss')
    plt.grid(True)
    plt.legend() 
    plt.savefig(args.save_dir + '/loss_curves.png')
    return 

if __name__ == "__main__":
    main()