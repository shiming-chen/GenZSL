
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from config import parse_args
from utils.helper_func import *
from utils.myLoader import MyDataloader
from model.myModels import Encoder, Decoder, Myclassifier


def generate_syn_data(generator, classes, embedding, in_package, args):
    syn_num = args.syn_num
    class_nums = classes.size(0)
    label2Idx = {value.item(): idx for idx, value in enumerate(classes)}
    syn_f, syn_l = [], []

    topk_l = in_package["topk_l"]
    seenlabel2Idx = in_package["sl2I"]
    label2features = in_package["l2f"]
    VAE_E = in_package["VAE_E"]
    for i in range(class_nums):
        iclass = classes[i]
        iclass_embedding = embedding[label2Idx[iclass.item()]]
        text_features = iclass_embedding.repeat(syn_num, 1)
        """
            - topk
        """
        topk_labels = topk_l[label2Idx[iclass.item()]]
        topk_idxes = torch.tensor([seenlabel2Idx[label.item()] for label in topk_labels])
        mix_topk_features = torch.zeros(text_features.size())
        weights = args.weights
        for i in range(args.topK):
            topi_features = label2features[topk_idxes[i]]
            n_samples = topi_features.size(0)
            n_rest = syn_num % n_samples
            features = topi_features.repeat(syn_num // n_samples, 1)
            if n_rest != 0:
                random_idxes = torch.randint(0, topi_features.size(0), (n_rest,))
                extra_features = topi_features[random_idxes]
                features = torch.cat((features, extra_features), dim=0)
            mix_topk_features += weights[i] * features.cpu()
        syn_input = mix_topk_features.to(args.device)

        with torch.no_grad():
           # ---------- VAE Encoder ---------- #
            mean, log_var = VAE_E(syn_input, text_features)
            std = torch.exp(0.5 * log_var)
            noise = torch.randn(mean.shape).to(args.device)
            z = std * noise + mean
            # ---------- VAE Decoder ---------- #
            syn_features = generator(torch.cat((z, text_features), dim=1))
            syn_features = syn_features / syn_features.norm(dim=-1, keepdim=True)
        syn_f.append(syn_features.cpu())
        syn_l.append(iclass.repeat(syn_num, 1))
    syn_f = np.concatenate(syn_f, axis=0)
    syn_l = np.concatenate(syn_l, axis=0)
    return syn_f, syn_l


args = parse_args()
print('args:', args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# ======================================== Prepare dataset ======================================== #
myloader = MyDataloader(args)
seenlabel2Idx = {value.item(): idx for idx, value in enumerate(myloader.seenclasses)}
seenidx2Label = {idx: value.item() for idx, value in enumerate(myloader.seenclasses)}
# ======================================== CLIP features ======================================== #
print(" ==> Load extracted features.")
CLIP_feature_path = args.image_root + f"/{args.dataset}/CLIP_feature.hdf5"
hf = h5py.File(CLIP_feature_path, 'r')
train_f, train_l = np.array(hf.get('train_f')), np.array(hf.get('train_l'))
testSeen_f, testSeen_l = np.array(hf.get('testSeen_f')), np.array(hf.get('testSeen_l'))
testUnseen_f, testUnseen_l = np.array(hf.get('testUnseen_f')), np.array(hf.get('testUnseen_l'))
# ---------- text description data ---------- #
seen_embeddings = np.array(hf.get('seen_embeddings'))
unseen_embeddings = np.array(hf.get('unseen_embeddings'))
print(" ====> Feature loaded.")
train_f = torch.from_numpy(train_f).float().to(args.device)
train_l = torch.from_numpy(train_l).float().to(args.device)
testSeen_f = torch.from_numpy(testSeen_f).float().to(args.device)
testSeen_l = torch.from_numpy(testSeen_l).float().to(args.device)
testUnseen_f = torch.from_numpy(testUnseen_f).float().to(args.device)
testUnseen_l = torch.from_numpy(testUnseen_l).float().to(args.device)
# ---------- text description data ---------- #
seen_embeddings = torch.from_numpy(seen_embeddings).float().to(args.device)
unseen_embeddings = torch.from_numpy(unseen_embeddings).float().to(args.device)

all_embeddings = torch.cat([seen_embeddings, unseen_embeddings], dim=0)
U, _, _ = np.linalg.svd(all_embeddings.cpu())
_U = U[:, 1:]
P_text = np.dot(_U, _U.T)
all_embeddings = torch.from_numpy(np.dot(P_text, all_embeddings.cpu())).float().to(args.device)
seen_embeddings = all_embeddings[:seen_embeddings.size(0)]
unseen_embeddings = all_embeddings[seen_embeddings.size(0):]

"""
    - calculate top k similarity
"""
# seen topk similarity
seenSimilarity = F.cosine_similarity(seen_embeddings.unsqueeze(1), seen_embeddings.unsqueeze(0), dim=2)
seentopk = np.argsort(-seenSimilarity.cpu(), axis=1)[:, 1:1+args.topK]  # except itself
for i in range(len(seentopk)):
    for j, idx in enumerate(seentopk[i]):
        seentopk[i][j] = seenidx2Label[idx.item()]
# unseen topk similarity
unseenSimilarity = F.cosine_similarity(unseen_embeddings.unsqueeze(1), seen_embeddings.unsqueeze(0), dim=2)
unseentopk = np.argsort(-unseenSimilarity.cpu(), axis=1)[:, :args.topK]
for i in range(len(unseentopk)):
    for j, idx in enumerate(unseentopk[i]):
        unseentopk[i][j] = seenidx2Label[idx.item()]
"""
    - for random get topk features from train_seen
    - label2features -> a list of features
"""
label_idxes = [torch.nonzero(train_l == label).squeeze() for label in myloader.seenclasses]
label2features = [train_f[idxes, :] for idxes in label_idxes]

# ======================================== Config models ======================================== #
netE_weight = f"{args.dataset}_netE_bestGZSL.pth"
netD_weight = f"{args.dataset}_netD_bestGZSL.pth"
netE = Encoder().to(args.device)
netD = Decoder().to(args.device)
netE.load_state_dict(torch.load(netE_weight, map_location='cpu'))
netD.load_state_dict(torch.load(netD_weight, map_location='cpu'))

netCLF = Myclassifier(seen_embeddings.size(1), len(myloader.allclasses_name)).to(args.device)
optimizerCLF = torch.optim.Adam(netCLF.parameters(), lr=1e-3, betas=(0.5, 0.999))
criterion = nn.CrossEntropyLoss().to(args.device)  # for classifier
# ======================================== Config dataloader ======================================== #
# ---------- trainloader ---------- #
trainset = TensorDataset(train_f, train_l)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
# ======================================== Test Stage ======================================== #

# ---------- synthesize unseen classes features ---------- #
syn_unseen_f = []
syn_unseen_l = []
with torch.no_grad():
    netD.eval()
    in_package = {"topk_l": unseentopk, "l2f": label2features, "VAE_E": netE, "sl2I": seenlabel2Idx}
    syn_unseen_f, syn_unseen_l = generate_syn_data(
        netD, myloader.unseenclasses, unseen_embeddings, in_package, args)
syn_unseen_f = torch.from_numpy(syn_unseen_f).float().to(args.device)
syn_unseen_l = torch.from_numpy(syn_unseen_l).squeeze(1).long().to(args.device)
print(" ==> feature synthesized.")

# ---------- train classifier ---------- #
new_train_f = torch.cat((train_f, syn_unseen_f), dim=0)
new_train_l = torch.cat((train_l, syn_unseen_l), dim=0)
new_trainset = TensorDataset(new_train_f, new_train_l)
new_trainloader = DataLoader(new_trainset, batch_size=args.batch_size, shuffle=False)
record = [0.0, 0.0, 0.0]

for clf_epoch in range(args.clf_nepoch):
    loss_clf = []
    for batch_features, batch_labels in new_trainloader:
        netCLF.train()
        netCLF.zero_grad()
        batch_features, batch_labels = batch_features.to(args.device), batch_labels.long().to(args.device)
        output = netCLF(batch_features)
        loss = criterion(output, batch_labels)
        loss_clf.append(loss.item())
        loss.backward()
        optimizerCLF.step()
        
    testloader = {"test_seen_f": testSeen_f, "test_seen_l": testSeen_l,
                    "test_unseen_f": testUnseen_f, "test_unseen_l": testUnseen_l}
    acc_train = eval_train_acc(myloader, new_train_f, new_train_l, netCLF, args.device)
    S, U, H, _ = eval_zs_gzsl(myloader, testloader, netCLF, args.device)
    print(f" ====> clf_epoch:{clf_epoch}; Loss:{sum(loss_clf)/len(loss_clf)}; Acc_train: {acc_train}")
    print(f" ====> S:{S}; U:{U}; H:{H}")

    if H > record[2]:
        record[:] = [S, U, H]


print('Best GZSL|CZSL => S:{:.6f}; U:{:.6f}; H:{:.6f}'.format(record[0], record[1], record[2]))