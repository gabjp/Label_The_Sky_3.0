import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint
from sklearn.metrics import f1_score, make_scorer, classification_report
import os
import argparse
import pickle
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

parser.add_argument('--rf_path',default = None, type=str)
parser.add_argument('--cnn_path', default=None, type=str)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device", flush=True)

FEATS = ["u_iso", "J0378_iso", "J0395_iso", "J0410_iso", "J0430_iso", "g_iso", "J0515_iso", "r_iso", "J0660_iso", "i_iso", "J0861_iso", "z_iso", "A", "B", "KRON_RADIUS", "FWHM_n", "w1mpro", "w2mpro", "target"]

def load_csv():
    df = pd.read_csv("data/all/clf_90_5_5.csv")
    df[['w1mpro', 'w2mpro']] = df[['w1mpro', 'w2mpro']].fillna(99)
    train = df[df.split == "train"][FEATS]
    test = df[df.split == "test"][FEATS]
    val = df[df.split == "val"][FEATS]
    test_wise = test[test.w1mpro != 99]
    test_nowise = test[test.w1mpro == 99]
    val_wise = val[val.w1mpro != 99]
    val_nowise = val[val.w1mpro == 99]

    return train, val_wise, val_nowise, test_wise, test_nowise

def main():
    args = parser.parse_args()
    print(args, flush=True)

    # Load CNN
    cnn = VGG16(3)
    checkpoint = torch.load(args.cnn_path)
    load_dict = checkpoint['model_state_dict']
    cnn.load_state_dict(load_dict, strict=False)
    cnn.to(device)
    cnn.eval()

    # Load RF
    with open(args.rf_path, 'rb') as file:
        rf = pickle.load(file)

    # Load data 
    
    _, val_loader, test_loader = get_loader("no_wise")
    _, _, val_nowise, _, test_nowise = load_csv()

    # Get preds
    
    image_list = []
    labels_list = []
    for image, label in val_loader:
        image_list.append(image)
        labels_list.append(label)
    images = torch.concat(image_list).to(device)
    labels = torch.concat(labels_list).to(device)
    out = cnn(images)
    cnn_pred_val = out.cpu().numpy()
    val_true = labels.cpu().numpy()

    image_list = []
    labels_list = []
    for image, label in test_loader:
        image_list.append(image)
        labels_list.append(label)
    images = torch.concat(image_list).to(device)
    labels = torch.concat(labels_list).to(device)
    out = cnn(images)
    cnn_pred_test = out.cpu().numpy()
    test_true = labels.cpu().numpy()

    rf_pred_val = rf.predict_proba(val_nowise.drop("target", axis=1))
    rf_pred_test = rf.predict_proba(test_nowise.drop("target", axis=1))

    # Grid search
    m = nn.Softmax(dim=0)

    print(m(cnn_pred_val))
    print(m(cnn_pred_test))
    print(rf_pred_val)
    print(rf_pred_test)

    # Print Results
    return 

if __name__ == "__main__":
    with torch.no_grad():
        main()