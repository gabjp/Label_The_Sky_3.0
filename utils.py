import torch 
from torch import nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
# Implementação da VGG16 com batch normalization e dropout.
# Fonte: https://blog.paperspace.com/vgg-from-scratch-pytorch/

class VGG16(nn.Module):
  def __init__(self, num_classes = 3, dropout = 0.3):
    super().__init__()

    self.features = nn.Sequential(
        nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride = 2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride = 2),
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride = 2),
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride = 2),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride = 2))

    # Camadas Fully Connected com menos parâmetros que a implementação original:

    self.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(1024, 512),
        nn.ReLU())

    self.fc1 = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(512, 512),
        nn.ReLU())

    self.fc2 = nn.Sequential(
        nn.Linear(512, num_classes))

  def forward(self, x):
    out = self.features(x)
    print(out.shape, flush=True)
    out = torch.reshape(out, (-1, 1024))
    out = self.fc(out)
    print(out.shape, flush=True)
    out = self.fc1(out)
    print(out.shape, flush=True)
    out = self.fc2(out)
    print(out.shape, flush=True)
    return out
  
def test(testloader, model, criterion, device):
  model.eval()
  epoch_loss = 0
  total_correct = 0
  total_samples = 0
  num_batch_count = 0

  for i, (image, label) in enumerate(testloader):
    image = image.to(device)
    label = label.to(device)

    out = model(image)
    loss = criterion(out, label)

    _, predicted = torch.max(out, 1)

    epoch_loss += loss.item()
    total_correct += (predicted == label).sum().item()
    total_samples += label.size(0)
    num_batch_count +=1

  loss = epoch_loss / num_batch_count
  acc = total_correct / total_samples * 100
  model.train()
  return loss, acc

def test_l(testloader, model, criterion, device):
  model.eval()
  epoch_loss = 0
  num_batch_count = 0

  for i, (image, label) in enumerate(testloader):
    image = image.to(device)
    label = label.to(device)

    out = model(image)
    loss = criterion(out, label)

    epoch_loss += loss.item()
    num_batch_count +=1

  loss = epoch_loss / num_batch_count
  model.train()
  return loss

class FromNpyDataset(torch.utils.data.Dataset):
  def __init__(self, X_path, y_path, is_unl = False, transform=None):
    super(FromNpyDataset, self).__init__()
    # store the raw tensors
    self.transform = transform
    
    self._x = np.load(X_path)
    if is_unl:
        self._y = np.load(y_path)[:,0:12]/35 
    else:  
        temp = np.load(y_path)
        self._y = np.zeros((temp.size, temp.max() + 1))
        self._y[np.arange(temp.size), temp] = 1

  def __len__(self):
    # a DataSet must know it size
    return self._x.shape[0]

  def __getitem__(self, index):
    x = self._x[index, :]
    y = self._y[index, :]
    x = self.transform(x).float()
    y = torch.Tensor(y).float()
    return x, y

def get_loader(dataset, batch_size=64):

    if dataset not in ["unl", "domain_1", "domain_2", "domain_3", "domain_4", "no_wise", "full"]:
       return None  
    
    if dataset != "full":
    
        if dataset == "unl":
            X_train_path = f"data/{dataset}/unl_w99_images_train.npy"
            y_train_path = f"data/{dataset}/unl_w99_tabular_train.npy"

            X_val_path = f"data/{dataset}/unl_w99_images_val.npy"
            y_val_path = f"data/{dataset}/unl_w99_tabular_val.npy"

            X_test_path = f"data/{dataset}/unl_w99_images_test.npy"
            y_test_path = f"data/{dataset}/unl_w99_tabular_test.npy"
        
        else:
            X_train_path = f"data/{dataset}/images_train.npy"
            y_train_path = f"data/{dataset}/classes_train.npy"

            X_val_path = f"data/{dataset}/images_val.npy"
            y_val_path = f"data/{dataset}/classes_val.npy"

            X_test_path = f"data/{dataset}/images_test.npy"
            y_test_path = f"data/{dataset}/classes_test.npy"

        train_set = FromNpyDataset(X_train_path, y_train_path, is_unl = dataset=="unl", transform=transforms.ToTensor())
        val_set = FromNpyDataset(X_val_path, y_val_path,  is_unl = dataset=="unl", transform=transforms.ToTensor())
        test_set = FromNpyDataset(X_test_path, y_test_path,  is_unl = dataset=="unl", transform=transforms.ToTensor())

    else:
        train_sets = []
        val_sets = []
        test_sets = []
        for domain in ["domain_1", "domain_2", "domain_3", "domain_4"]:
            X_train_path = f"data/{dataset}/images_train.npy"
            y_train_path = f"data/{dataset}/classes_train.npy"

            X_val_path = f"data/{dataset}/images_val.npy"
            y_val_path = f"data/{dataset}/classes_val.npy"

            X_test_path = f"data/{dataset}/images_test.npy"
            y_test_path = f"data/{dataset}/classes_test.npy"

            train_sets.append(FromNpyDataset(X_train_path, y_train_path, is_unl = dataset=="unl",  transform=transforms.ToTensor()))
            val_sets.append(FromNpyDataset(X_val_path, y_val_path,  is_unl = dataset=="unl", transform=transforms.ToTensor()))
            test_sets.append(FromNpyDataset(X_test_path, y_test_path,  is_unl = dataset=="unl", transform=transforms.ToTensor()))
        
        train_set = ConcatDataset(train_sets)
        val_set = ConcatDataset(val_sets)
        test_set = ConcatDataset(test_sets)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader