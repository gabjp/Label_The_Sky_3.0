import argparse
import torch
from utils import get_loader, VGG16, test, compute_f1_score
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import numpy as np


parser = argparse.ArgumentParser(description='Pretrain magnitude task')

parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=0.0007, type=float, help='weight decay')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout')
parser.add_argument('--optimizer', default="adam", type=str, help="otimizer algorithm")
parser.add_argument('--data', default="no_wise", type=str, help="dataset")
parser.add_argument('--checkpoint_path', type=str, help="pretrained path")

parser.add_argument('--w_lr', default=1e-4, type=float, help='warmup learning rate')
parser.add_argument('--w_momentum', default=0.9, type=float, help='warmup momentum')
parser.add_argument('--w_weight_decay', default=0.0007, type=float, help='warmup weight decay')
parser.add_argument('--w_epochs', default=10, type=int, help='warmup number of total epochs to run')
parser.add_argument('--w_dropout', default=0.3, type=float, help='warmup dropout')
parser.add_argument('--w_optimizer', default="adam", type=str, help="warmup otimizer algorithm")

class CustomMAE(nn.Module):
    def __init__(self):
        super(CustomMAE, self).__init__()
        self.mae = nn.L1Loss()

    def forward(self, inputs, targets):
        mvalue = 99/35
        mask = torch.tensor(targets != mvalue, dtype=torch.float32)
        return self.mae(inputs*mask, targets*mask)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device", flush=True)

def main():
    args = parser.parse_args()
    print(args, flush=True)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    #load data
    train_loader, val_loader, test_loader = get_loader(args.data, args.batch_size)

    #load pretrained model
    model = VGG16(3, args.dropout)
    checkpoint = torch.load(args.checkpoint_path)
    load_dict =  {k: checkpoint['model_state_dict'][k] for k in checkpoint['model_state_dict'].keys() if 'features' in k}
    model.load_state_dict(load_dict, strict=False)

    #train - WARMUP

    weights = torch.Tensor(compute_class_weight(class_weight='balanced', classes=[0,1,2], y=np.argmax(train_loader.dataset._y, axis=1))).to(device)
    print(f"Weights: {weights}")

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    w_groups = [{'params': model.fc.parameters()},{'params':model.fc1.parameters()},{'params':model.fc2.parameters()}]

    if args.w_optimizer == "adam":
        opt = torch.optim.Adam(w_groups, lr=args.w_lr, weight_decay=args.w_weight_decay)
    else:
        opt = torch.optim.SGD(w_groups, lr=args.w_lr, momentum=args.w_momentum, weight_decay=args.w_weight_decay)

    model.train()

    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []
    test_fscore = []
    best_fscore = 0

    for n_epoch in range(args.w_epochs):
        epoch_train_loss = 0
        total_correct = 0
        total_samples = 0
        num_batch_count = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            opt.zero_grad()
            out = model(images)

            loss = criterion(out, labels)
            loss.backward()
            opt.step()

            _, predicted = torch.max(out, 1)

            epoch_train_loss += loss.item()
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            num_batch_count +=1

        print(f"[{n_epoch+1}/{args.w_epochs}] - Training loss: {epoch_train_loss/num_batch_count} - Training accuracy: {total_correct / total_samples * 100}")
        train_loss.append(epoch_train_loss/num_batch_count)
        train_acc.append(total_correct / total_samples * 100)

        val_loss, val_acc = test(val_loader,model, criterion, device)
        print(f"[{n_epoch+1}/{args.w_epochs}] - Validation loss: {val_loss} - Validation accuracy: {val_acc}")
        test_loss.append(val_loss)
        test_acc.append(val_acc)

        test_fscore.append(compute_f1_score(val_loader, model, device )*100)
        print(f"[{n_epoch+1}/{args.w_epochs}] - Validation f1: {test_fscore[-1]} ")

        if test_fscore[-1] > best_fscore:
            best_fscore = test_fscore[-1]
            torch.save({
                'epoch': n_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': criterion,
                }, args.save_dir + "/checkpoint.pth")

    #train

    if args.w_optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for n_epoch in range(args.epochs):
        epoch_train_loss = 0
        total_correct = 0
        total_samples = 0
        num_batch_count = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            opt.zero_grad()
            out = model(images)

            loss = criterion(out, labels)
            loss.backward()
            opt.step()

            _, predicted = torch.max(out, 1)

            epoch_train_loss += loss.item()
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            num_batch_count +=1

        print(f"[{n_epoch+1}/{args.epochs}] - Training loss: {epoch_train_loss/num_batch_count} - Training accuracy: {total_correct / total_samples * 100}")
        train_loss.append(epoch_train_loss/num_batch_count)
        train_acc.append(total_correct / total_samples * 100)

        val_loss, val_acc = test(val_loader,model, criterion, device)
        print(f"[{n_epoch+1}/{args.w_epochs}] - Validation loss: {val_loss} - Validation accuracy: {val_acc}")
        test_loss.append(val_loss)
        test_acc.append(val_acc)

        test_fscore.append(compute_f1_score(val_loader, model, device )*100)
        print(f"[{n_epoch+1}/{args.epochs}] - Validation f1: {test_fscore[-1]} ")

        if test_fscore[-1] > best_fscore:
            best_fscore = test_fscore[-1]
            torch.save({
                'epoch': n_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': criterion,
                }, args.save_dir + "/checkpoint.pth")

    #save checkpoint and figures

    plt.figure(figsize=(8, 6))
    plt.plot(list(range(args.epochs + args.w_epochs)), test_loss, marker='o', linestyle='-', label='Validation Loss')
    plt.title('Validation and Training Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(list(range(args.epochs + args.w_epochs)), train_loss, marker='o', linestyle='-', label='Training Loss')
    plt.grid(True)
    plt.legend() 
    plt.savefig(args.save_dir + '/loss_curves.png')
    plt.clf()

    plt.figure(figsize=(8, 6))
    plt.plot(list(range(args.epochs + args.w_epochs)), test_acc, marker='o', linestyle='-', label='Validation Accuracy')
    plt.title('Validation and Training Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.plot(list(range(args.epochs + args.w_epochs)), train_acc, marker='o', linestyle='-', label='Training Accuracy')
    plt.plot(list(range(args.epochs + args.w_epochs)), test_fscore, marker='o', linestyle='-', label='Validation F-score')
    plt.grid(True)
    plt.legend() 
    plt.savefig(args.save_dir + '/metrics_curves.png')

    #get validation/test metrics
    model = VGG16(3, args.dropout)
    checkpoint = torch.load(args.checkpoint_path)
    load_dict =  checkpoint["model_state_dict"]
    model.load_state_dict(load_dict) 
    model.eval()

    for loader, name in zip((val_loader, test_loader), ("Validation:", "Test:")):
        images = torch.Tensor(loader.dataset._x).to(device)
        out = model(images)
        _, predicted = torch.max(out, 1).numpy()
        true = loader.dataset._y
        print(name)
        print(classification_report(true, predicted, labels = ["QSO", "STAR", "GAL"]))
    # CHANGE TO EVAL MODE
    return

if __name__ == "__main__":
    main()