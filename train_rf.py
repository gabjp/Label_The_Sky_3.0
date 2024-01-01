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

parser = argparse.ArgumentParser(description='Pretrain magnitude task')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)

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

def hp_search(X, y):

    rf = RandomForestClassifier()

    param_dist = {
        'n_estimators': randint(50, 500),
        'max_depth': [None] + list(randint(5, 50).rvs(4)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20)
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    f1_scorer = make_scorer(f1_score, average='micro')

    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=10, cv=skf, scoring=f1_scorer, random_state=42)
    random_search.fit(X, y)

    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print("Best Hyperparameters:", best_params)
    print("Best Score (F1 score):", best_score)

    return best_params

def main():
    args = parser.parse_args()

    train, val_wise, val_nowise, test_wise, test_nowise = load_csv()
    best_params = hp_search(train.drop("target", axis=1), train["target"])

    rf = RandomForestClassifier(**best_params,random_state=42) # add hyperparameters
    rf.fit(train.drop("target", axis=1), train["target"])

    wise_pred_val = rf.predict(val_wise.drop("target", axis=1))
    nowise_pred_val = rf.predict(val_nowise.drop("target", axis=1))
    wise_pred_test = rf.predict(test_wise.drop("target", axis=1))
    nowise_pred_test = rf.predict(test_nowise.drop("target", axis=1))

    print("WISE VAL")
    print(classification_report(val_wise["target"], wise_pred_val),digits = 4, target_names = ["QSO", "STAR", "GAL"])
    print("NO WISE VAL")
    print(classification_report(val_nowise["target"], nowise_pred_val),digits = 4, target_names = ["QSO", "STAR", "GAL"])
    print("WISE TEST")
    print(classification_report(test_wise["target"], wise_pred_test),digits = 4, target_names = ["QSO", "STAR", "GAL"])
    print("NO WISE TEST")
    print(classification_report(test_nowise["target"], nowise_pred_test),digits = 4, target_names = ["QSO", "STAR", "GAL"])

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open(args.save_dir + '/model.pkl', 'wb') as file:
        pickle.dump(rf, file)

    return

if __name__ == "__main__":
    main()