import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "../../Label_The_Sky_2.0/Data/ready/clf/"
csv = pd.read_csv("./all/clf_90_5_5.csv")

nowise = csv[csv['w1mpro'].isna()]
dist1 = csv.dropna().sort_values("r_iso").iloc[0:27465]
dist2 = csv.dropna().sort_values("r_iso").iloc[27465:54930]
dist3 = csv.dropna().sort_values("r_iso").iloc[54930:82395]
dist4 = csv.dropna().sort_values("r_iso").iloc[82395:]

nowise_train = list(nowise[nowise.split == "train"]["inside_split_order"])
dist1_train = list(dist1[dist1.split == "train"]["inside_split_order"])
dist2_train = list(dist2[dist2.split == "train"]["inside_split_order"])
dist3_train = list(dist3[dist3.split == "train"]["inside_split_order"])
dist4_train = list(dist4[dist4.split == "train"]["inside_split_order"])

nowise_val = list(nowise[nowise.split == "val"]["inside_split_order"])
dist1_val = list(dist1[dist1.split == "val"]["inside_split_order"])
dist2_val = list(dist2[dist2.split == "val"]["inside_split_order"])
dist3_val = list(dist3[dist3.split == "val"]["inside_split_order"])
dist4_val = list(dist4[dist4.split == "val"]["inside_split_order"])

nowise_test = list(nowise[nowise.split == "test"]["inside_split_order"])
dist1_test = list(dist1[dist1.split == "test"]["inside_split_order"])
dist2_test = list(dist2[dist2.split == "test"]["inside_split_order"])
dist3_test = list(dist3[dist3.split == "test"]["inside_split_order"])
dist4_test = list(dist4[dist4.split == "test"]["inside_split_order"])

print("Loaded csv", flush=True)

class_train = np.load(DATA_PATH + "clf_90_5_5_class_train.npy")
images_train = np.load(DATA_PATH + "clf_90_5_5_images_train.npy")
tabular_train = np.load(DATA_PATH + "clf_90_5_5_tabular_train.npy")

print("Loaded train", flush=True)

class_val = np.load(DATA_PATH + "clf_90_5_5_class_val.npy")
images_val = np.load(DATA_PATH + "clf_90_5_5_images_val.npy")
tabular_val = np.load(DATA_PATH + "clf_90_5_5_tabular_val.npy")

print("Loaded val", flush=True)

class_test = np.load(DATA_PATH + "clf_90_5_5_class_test.npy")
images_test = np.load(DATA_PATH + "clf_90_5_5_images_test.npy")
tabular_test = np.load(DATA_PATH + "clf_90_5_5_tabular_test.npy")

print("Loaded test", flush=True)

############################ NO_WISE SPLITS #########################

nowise_tabular_train = tabular_train[nowise_train]
nowise_class_train = class_train[nowise_train]
nowise_images_train = images_train[nowise_train]

nowise_tabular_val = tabular_val[nowise_val]
nowise_class_val = class_val[nowise_val]
nowise_images_val = images_val[nowise_val]

nowise_tabular_test = tabular_test[nowise_test]
nowise_class_test = class_test[nowise_test]
nowise_images_test = images_test[nowise_test]

np.save("no_wise/tabular_train.npy", nowise_tabular_train)
np.save("no_wise/class_train.npy", nowise_class_train)
np.save("no_wise/images_train.npy", nowise_images_train)

np.save("no_wise/tabular_val.npy", nowise_tabular_val)
np.save("no_wise/class_val.npy", nowise_class_val)
np.save("no_wise/images_val.npy", nowise_images_val)

np.save("no_wise/tabular_test.npy", nowise_tabular_test)
np.save("no_wise/class_test.npy", nowise_class_test)
np.save("no_wise/images_test.npy", nowise_images_test)

print("Loaded no_wise", flush=True)
############################ DIST_1 SPLITS #########################

dist1_tabular_train = tabular_train[dist1_train]
dist1_class_train = class_train[dist1_train]
dist1_images_train = images_train[dist1_train]

dist1_tabular_val = tabular_val[dist1_val]
dist1_class_val = class_val[dist1_val]
dist1_images_val = images_val[dist1_val]

dist1_tabular_test = tabular_test[dist1_test]
dist1_class_test = class_test[dist1_test]
dist1_images_test = images_test[dist1_test]

np.save("domain_1/tabular_train.npy", dist1_tabular_train)
np.save("domain_1/class_train.npy", dist1_class_train)
np.save("domain_1/images_train.npy", dist1_images_train)

np.save("domain_1/tabular_val.npy", dist1_tabular_val)
np.save("domain_1/class_val.npy", dist1_class_val)
np.save("domain_1/images_val.npy", dist1_images_val)

np.save("domain_1/tabular_test.npy", dist1_tabular_test)
np.save("domain_1/class_test.npy", dist1_class_test)
np.save("domain_1/images_test.npy", dist1_images_test)

print("Loaded domain 1", flush=True)
############################ DIST_2 SPLITS #########################

dist2_tabular_train = tabular_train[dist2_train]
dist2_class_train = class_train[dist2_train]
dist2_images_train = images_train[dist2_train]

dist2_tabular_val = tabular_val[dist2_val]
dist2_class_val = class_val[dist2_val]
dist2_images_val = images_val[dist2_val]

dist2_tabular_test = tabular_test[dist2_test]
dist2_class_test = class_test[dist2_test]
dist2_images_test = images_test[dist2_test]

np.save("domain_2/tabular_train.npy", dist2_tabular_train)
np.save("domain_2/class_train.npy", dist2_class_train)
np.save("domain_2/images_train.npy", dist2_images_train)

np.save("domain_2/tabular_val.npy", dist2_tabular_val)
np.save("domain_2/class_val.npy", dist2_class_val)
np.save("domain_2/images_val.npy", dist2_images_val)

np.save("domain_2/tabular_test.npy", dist2_tabular_test)
np.save("domain_2/class_test.npy", dist2_class_test)
np.save("domain_2/images_test.npy", dist2_images_test)

print("Loaded domain 2", flush=True)
############################ DIST_3 SPLITS #########################

dist3_tabular_train = tabular_train[dist3_train]
dist3_class_train = class_train[dist3_train]
dist3_images_train = images_train[dist3_train]

dist3_tabular_val = tabular_val[dist3_val]
dist3_class_val = class_val[dist3_val]
dist3_images_val = images_val[dist3_val]

dist3_tabular_test = tabular_test[dist3_test]
dist3_class_test = class_test[dist3_test]
dist3_images_test = images_test[dist3_test]

np.save("domain_3/tabular_train.npy", dist3_tabular_train)
np.save("domain_3/class_train.npy", dist3_class_train)
np.save("domain_3/images_train.npy", dist3_images_train)

np.save("domain_3/tabular_val.npy", dist3_tabular_val)
np.save("domain_3/class_val.npy", dist3_class_val)
np.save("domain_3/images_val.npy", dist3_images_val)

np.save("domain_3/tabular_test.npy", dist3_tabular_test)
np.save("domain_3/class_test.npy", dist3_class_test)
np.save("domain_3/images_test.npy", dist3_images_test)

print("Loaded domain 3", flush=True)
############################ DIST_4 SPLITS #########################

dist4_tabular_train = tabular_train[dist4_train]
dist4_class_train = class_train[dist4_train]
dist4_images_train = images_train[dist4_train]

dist4_tabular_val = tabular_val[dist4_val]
dist4_class_val = class_val[dist4_val]
dist4_images_val = images_val[dist4_val]

dist4_tabular_test = tabular_test[dist4_test]
dist4_class_test = class_test[dist4_test]
dist4_images_test = images_test[dist4_test]

np.save("domain_4/tabular_train.npy", dist4_tabular_train)
np.save("domain_4/class_train.npy", dist4_class_train)
np.save("domain_4/images_train.npy", dist4_images_train)

np.save("domain_4/tabular_val.npy", dist4_tabular_val)
np.save("domain_4/class_val.npy", dist4_class_val)
np.save("domain_4/images_val.npy", dist4_images_val)

np.save("domain_4/tabular_test.npy", dist4_tabular_test)
np.save("domain_4/class_test.npy", dist4_class_test)
np.save("domain_4/images_test.npy", dist4_images_test)

print("Loaded domain 4", flush=True)