import torch
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim
from train_model import *
from util import *
from dataset import *
from ot_util import *
from model import *
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_pseudo_labels(dataloader, model, confidence_q=0.1):
    logits = []
    model.eval()
    with torch.no_grad():
        for x in dataloader:
            if len(x) == 3:
                data, _, _ = x
            else:
                data, _ = x
            data = data.to(device)
            logits.append(model(data))
    
    logits = torch.cat(logits)
    confidence = torch.max(logits, dim=1)[0] - torch.min(logits, dim=1)[0]
    alpha = torch.quantile(confidence, confidence_q)
    indices = torch.where(confidence >= alpha)[0].to("cpu")
    labels = torch.argmax(logits, axis=1) #[indices]
    
    return labels.cpu().detach().type(torch.int64), list(indices.detach().numpy())


def self_train(args, source_model, datasets, epochs=10):
    steps = len(datasets)
    teacher = source_model
    targetset = datasets[-1]
        
    targetloader = DataLoader(targetset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("------------Direct adapt performance----------")
    direct_acc = test(targetloader, teacher)

    # start self-training on intermediate domains
    for i in range(steps):
        print(f"--------Training on the {i}th domain--------")
        trainset = datasets[i]
        ogloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                
        test(targetloader, teacher)
        train_labs, train_idx = get_pseudo_labels(ogloader, teacher)

        if torch.is_tensor(trainset.data):
            data = trainset.data.cpu().detach().numpy()
        else:
            data = trainset.data
        trainset  = EncodeDataset(data, train_labs, trainset.transform)
        
        # filter out the least 10% confident data
        filter_trainset = Subset(trainset, train_idx)
        print("Trainset size: " + str(len(filter_trainset)))

        trainloader =  DataLoader(filter_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        # initialize and train student model
        student = copy.deepcopy(teacher)
        optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-4)

        for i in range(1, epochs+1):
            train(i, trainloader, student, optimizer)
            if i % 5 == 0:
                 test(targetloader, student)
        print("------------Performance on the current domain----------")
        test(trainloader, student)

        # test on the target domain
        print("------------Performance on the target domain----------")
        st_acc = test(targetloader, student)

        teacher = copy.deepcopy(student)
    
    return direct_acc, st_acc

