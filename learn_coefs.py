import torch
import sys
sys.path.append("models")
sys.path.append(".")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
from functorch import make_functional_with_buffers, vmap, grad
from utils import get_loader, VGG16
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight



# 1) Read parameters

parser = argparse.ArgumentParser(description='MAML adaptation to learn coefs')

parser.add_argument('--inner-lr', type=float, default=1e-3) # Inner loop learning rate
parser.add_argument('--outer-lr', type=float, default=1e-4) # Outer loop learning rate
parser.add_argument('--inner-steps', type=int, default=10) # Inner loop steps
parser.add_argument('--outer-steps', type=int, default=60) # Outer loop steps
parser.add_argument('--batch-size', type=int, default=64) # Number of examples to compute gradient
parser.add_argument('--dataset', default='no_wise', type=str) # Dataset
parser.add_argument('--workers', default=2, type=int) # Dataset loading workers
parser.add_argument('--reinit-fc', action='store_true')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--runs', default=1, type=int)
parser.add_argument('--weights', type=str, default='0.25,0.25,0.25,0.25')
parser.add_argument('--inner-momentum', type=float, default=0.9) # Inner loop momentum
parser.add_argument('--outer-momentum', type=float, default=0.9) # Outer loop momentum
parser.add_argument('--permutation', type=str, default='0,1,2,3')
parser.add_argument('--dropout', type=float, default=0.3)



def accuracy(func, combined_weights, buffers, val_loader):
    func.eval()
    total_correct = 0
    total_instances = 0

    for i, (image, label) in enumerate(val_loader):

        image = image.to(device)
        label = label.to(device)

        output = func(combined_weights, buffers, image)
        classifications = torch.argmax(output, dim=1)
        correct_predictions = sum(classifications==label).item()
        total_correct+=correct_predictions
        total_instances+=len(image)
    
    acc = round(total_correct/total_instances, 4) * 100
    func.train()
    return acc

def compute_f1_score(func, combined_weights, buffers, loader):
    func.eval()
    image_list = []
    labels_list = []
    for image, label in loader:
        image_list.append(image)
        labels_list.append(label)
    images = torch.concat(image_list).to(device)
    labels = torch.concat(labels_list).to(device)
    out = func(combined_weights, buffers, images)
    _, predicted = torch.max(out, 1)
    _, true = torch.max(labels, 1)
    func.train()
    return f1_score(true.cpu().numpy(), predicted.cpu().numpy(), average='macro')

def main():
    args = parser.parse_args()
    print(args)
    
    # Load Data:

    train_loader, val_loader, _ = get_loader(args.dataset, args.batch_size)
    
    # Load Models:

    models = []
    permutation = list(map(int, args.permutation.split(",")))

    domains = [0,0,0,0]
    domains[0] = torch.load("experiments/finetune_domain_1/checkpoint.pth")["model_state_dict"]
    domains[1] = torch.load("experiments/finetune_domain_2/checkpoint.pth")["model_state_dict"]
    domains[2] = torch.load("experiments/finetune_domain_3/checkpoint.pth")["model_state_dict"]
    domains[3]= torch.load("experiments/finetune_domain_4/checkpoint.pth")["model_state_dict"]
        
    for i in range(4):
        model = VGG16(dropout=args.dropout).cuda()
        model.load_state_dict(domains[i])
        model.to(device)
        models.append(model)

    train_model = VGG16(dropout=args.dropout)
    train_model.to(device)

    print('load success') 
    
    # 3) MAML
    weights = torch.Tensor(compute_class_weight(class_weight='balanced', classes=[0,1,2], y=np.argmax(train_loader.dataset._y, axis=1))).to(device)
    _, p1, buffer_1 = make_functional_with_buffers(models[permutation[0]])
    _, p2, buffer_2 = make_functional_with_buffers(models[permutation[1]])
    _, p3, buffer_3 = make_functional_with_buffers(models[permutation[2]])
    _, p4, buffer_4 = make_functional_with_buffers(models[permutation[3]])
    func, params, buffers = make_functional_with_buffers(train_model)
    n_layers = len(p1)

    criterion = nn.CrossEntropyLoss(weight=weights) 

    func.train()

    best_f1 = 0
    best_coefs = None

    for r in range(args.runs):

        print(f"Starting Run {r+1}")

        if args.weights == "random":
            coefs = torch.softmax(torch.rand(4, device=device, requires_grad=True), dim=0)

        else:
            coefs = torch.tensor(list(map(float, args.weights.split(","))), device=device, requires_grad=True)

        print(coefs)

        for o_step in range(args.outer_steps):
            # Reset weights
            print(f"Outer step ({o_step}/{args.outer_steps}): {coefs}", flush=True)

            combined_weights = [coefs[0]* p1[k]+ coefs[1]*p2[k] + coefs[2]*p3[k] + coefs[3]*p4[k] for k in range(n_layers)]

            total_epochs = 0
            inner_step = 0
            while inner_step < args.inner_steps + 1:
                for image, label in train_loader:

                    if inner_step == 0: #Sample data for meta update
                        outer_image = image.to(device)
                        outer_label = label.to(device)
                        inner_step += 1
                        continue

                    if inner_step == args.inner_steps + 1: 
                        print("Number of inner steps reached")
                        break

                    image = image.to(device)
                    label = label.to(device)
                                                    
                    outputs = func(combined_weights, buffers, image)
                    loss = criterion(outputs, label)

                    # Manually compute gradient 
                    grad = torch.autograd.grad(loss, combined_weights, create_graph=True)

                    # Momentum
                    if inner_step == 1:
                        inner_update = grad
                    else:
                        inner_update = tuple((args.inner_momentum * iu + g) for iu,g in zip(inner_update, grad))

                    #Update weights
                    combined_weights = tuple((w - args.inner_lr * iu for w,iu in zip(combined_weights, inner_update) ))

                    inner_step += 1

                total_epochs += 1

            print(f"Epochs finished: {total_epochs - 1}")


            outputs = func(combined_weights, buffers, outer_image)
            loss = criterion(outputs, outer_label)

            with torch.no_grad():
                grad = torch.autograd.grad(loss, coefs, create_graph=False)

                #acc = accuracy(func, combined_weights, buffers, val_loader)
                f1 = compute_f1_score(func, combined_weights, buffers, val_loader)
                if f1 > best_f1:
                    best_f1 = f1
                    best_coefs = coefs.detach().clone()

                #print(f"Inner loop validation accuracy: {round(acc,2)}")
                print(f"Inner loop validation f1: {round(f1,4)*100}")

                if o_step == 0:
                    outer_update = grad[0]
                else:
                    outer_update = args.outer_momentum * outer_update + grad[0]
            
                print(f"outer gradient: {grad[0]}")
                print(f"outer update: {outer_update}")    
                coefs = coefs - args.outer_lr * outer_update
            
            coefs.requires_grad_(True)
            print()

    print(f"Final coefs: {best_coefs}")
    s_coefs = torch.softmax(best_coefs, dim=0)
    print(f"Softmax coefs: {s_coefs}")
    n_coefs = best_coefs / torch.sum(best_coefs)
    print(f"Normalized coefs: {n_coefs}")
    print(f"Inner loop accuracy: {best_acc}")

if __name__ == "__main__":
    main()