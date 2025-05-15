
"""
    - Using topk seen samples' features
    - get a list [(sample_nums1, f_dim), (sample_nums2, f_dim), ...]
"""

import os
import clip
import h5py
import time
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

from config import parse_args
from utils.helper_func import *
from utils.myLoader import MyDataloader
from model.myModels import Encoder, Decoder, Myclassifier


def create_unique_folder_name(base_folder_path):
    count = 0
    new_folder_name = base_folder_path
    while os.path.exists(new_folder_name):
        count += 1
        new_folder_name = f"{base_folder_path}({count})"
    return new_folder_name


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
        topk_idxes = torch.tensor([seenlabel2Idx[label.item()]
                                  for label in topk_labels])  # (1, topk)
        mix_topk_features = torch.zeros(text_features.size())  # init
        weights = args.weights
        for i in range(args.topK):
            topi_features = label2features[topk_idxes[i]]
            n_samples = topi_features.size(0)
            n_rest = syn_num % n_samples
            features = topi_features.repeat(syn_num // n_samples, 1)
            if n_rest != 0:
                random_idxes = torch.randint(
                    0, topi_features.size(0), (n_rest,))
                extra_features = topi_features[random_idxes]
                features = torch.cat((features, extra_features), dim=0)
            mix_topk_features += weights[i] * features.cpu()
        syn_input = mix_topk_features.to(args.device)
        #########################################

        with torch.no_grad():
           # ---------- VAE Encoder ---------- #
            mean, log_var = VAE_E(syn_input, text_features)
            std = torch.exp(0.5 * log_var)
            noise = torch.randn(mean.shape).to(args.device)
            z = std * noise + mean
            # ---------- VAE Decoder ---------- #
            syn_features = generator(torch.cat((z, text_features), dim=1))
            syn_features = syn_features / \
                syn_features.norm(dim=-1, keepdim=True)
        syn_f.append(syn_features.cpu())
        syn_l.append(iclass.repeat(syn_num, 1))
    syn_f = np.concatenate(syn_f, axis=0)
    syn_l = np.concatenate(syn_l, axis=0)
    return syn_f, syn_l


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ======================================== Log ======================================== #
    os.makedirs(args.log_root_path, exist_ok=True)
    outlogDir = "{}/{}".format(args.log_root_path, args.dataset)
    os.makedirs(outlogDir, exist_ok=True)
    num_exps = len([f.path for f in os.scandir(outlogDir) if f.is_dir()])
    outlogPath = os.path.join(
        outlogDir, create_unique_folder_name(outlogDir + f"/exp{num_exps}"))
    os.makedirs(outlogPath, exist_ok=True)
    t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
    log = outlogPath + "/" + t + '.txt'
    logging.basicConfig(format='%(message)s', level=logging.INFO,
                        filename=log,
                        filemode='w')
    logger = logging.getLogger(__name__)
    argsDict = args.__dict__
    for eachArg, value in argsDict.items():
        logger.info(eachArg + ':' + str(value))
    logger.info("="*50)

    # ======================================== CLIP ======================================== #
    clip_model, _ = clip.load(args.backbone)
    clip_model.to(args.device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    # ======================================== Prepare dataset ======================================== #
    myloader = MyDataloader(args)
    # maps
    seenlabel2Idx = {value.item(): idx for idx,
                     value in enumerate(myloader.seenclasses)}
    seenidx2Label = {idx: value.item()
                     for idx, value in enumerate(myloader.seenclasses)}
    # ======================================== CLIP features ======================================== #
    print(" ==> Load extracted features.")
    CLIP_feature_path = args.image_root + f"/{args.dataset}/CLIP_feature.hdf5"
    hf = h5py.File(CLIP_feature_path, 'r')
    # ---------- train data ---------- #
    train_f, train_l = np.array(hf.get('train_f')), np.array(hf.get('train_l'))
    # ---------- test unseen data ---------- #
    testUnseen_f, testUnseen_l = np.array(
        hf.get('testUnseen_f')), np.array(hf.get('testUnseen_l'))
    # ---------- text description data ---------- #
    seen_embeddings = np.array(hf.get('seen_embeddings'))
    unseen_embeddings = np.array(hf.get('unseen_embeddings'))
    print(" ====> Feature loaded.")
    # ---------- train data ---------- #
    train_f = torch.from_numpy(train_f).float().to(args.device)
    train_l = torch.from_numpy(train_l).float().to(args.device)
    # ---------- test unseen data ---------- #
    testUnseen_f = torch.from_numpy(testUnseen_f).float().to(args.device)
    testUnseen_l = torch.from_numpy(testUnseen_l).float().to(args.device)
    _, testUnseen_l, _ = torch.unique(testUnseen_l, return_inverse=True, return_counts=True)
    # ---------- text description data ---------- #
    seen_embeddings = torch.from_numpy(seen_embeddings).float().to(args.device)
    unseen_embeddings = torch.from_numpy(
        unseen_embeddings).float().to(args.device)
    # print(orig_f_train.shape, topk_f_train.shape, labels_train.shape)

    """
        - svd
    """
    if args.use_svd:
        all_embeddings = torch.cat([seen_embeddings, unseen_embeddings], dim=0)
        U, _, _ = np.linalg.svd(all_embeddings.cpu())
        _U = U[:, 1:]
        P_text = np.dot(_U, _U.T)
        all_embeddings = torch.from_numpy(
            np.dot(P_text, all_embeddings.cpu())).float().to(args.device)
        seen_embeddings = all_embeddings[:seen_embeddings.size(0)]
        unseen_embeddings = all_embeddings[seen_embeddings.size(0):]

    """
        - calculate top k similarity
    """
    # seen topk similarity
    seenSimilarity = F.cosine_similarity(
        seen_embeddings.unsqueeze(1), seen_embeddings.unsqueeze(0), dim=2)
    seentopk = np.argsort(-seenSimilarity.cpu(),
                          axis=1)[:, 1:1+args.topK]  # except itself
    for i in range(len(seentopk)):
        for j, idx in enumerate(seentopk[i]):
            seentopk[i][j] = seenidx2Label[idx.item()]
    # unseen topk similarity
    unseenSimilarity = F.cosine_similarity(
        unseen_embeddings.unsqueeze(1), seen_embeddings.unsqueeze(0), dim=2)
    unseentopk = np.argsort(-unseenSimilarity.cpu(), axis=1)[:, :args.topK]
    for i in range(len(unseentopk)):
        for j, idx in enumerate(unseentopk[i]):
            unseentopk[i][j] = seenidx2Label[idx.item()]
    """
        - for random get topk features from train_seen
        - label2features -> a list of features
    """
    label_idxes = [torch.nonzero(train_l == label).squeeze()
                   for label in myloader.seenclasses]
    label2features = [train_f[idxes, :] for idxes in label_idxes]

    # ======================================== Config models ======================================== #
    netE = Encoder().to(args.device)
    netD = Decoder().to(args.device)
    optimizerE = torch.optim.Adam(
        netE.parameters(), lr=1e-3, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(
        netD.parameters(), lr=1e-3, betas=(0.5, 0.999))
    netCLF = Myclassifier(seen_embeddings.size(1), len(
        myloader.unseen_names)).to(args.device)
    optimizerCLF = torch.optim.Adam(
        netCLF.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.CrossEntropyLoss().to(args.device)  # for classifier
    best_Epoch, best_CZSL = 0, 0.0
    # ======================================== Config dataloader ======================================== #
    # ---------- trainloader ---------- #
    trainset = TensorDataset(train_f, train_l)
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False)
    # ======================================== Main pipeline ======================================== #
    early_break = 0
    for epoch in range(1, args.nepoch+1):
        # -------------------- Train Stage -------------------- #
        for _ in range(0, args.loop):
            netE.train()
            netD.train()
            loss_list = []
            print(' ==> Train Epoch: {:}/{:}'.format(epoch, args.nepoch))
            for batch_features, batch_labels in trainloader:
                batch_features, batch_labels = batch_features.to(
                    args.device), batch_labels.to(args.device)
                batch_idxes = torch.tensor(
                    [seenlabel2Idx[label.item()] for label in batch_labels])
                text_features = seen_embeddings[batch_idxes].float().to(
                    args.device)

                """
                    - topk
                """
                batch_topk_labels = seentopk[batch_idxes]
                batch_topk_idxes = []
                for topk_labels in batch_topk_labels:
                    idxes = [seenlabel2Idx[label.item()] for label in topk_labels]
                    batch_topk_idxes.append(idxes)
                batch_topk_idxes = np.array(batch_topk_idxes).reshape(
                    (batch_features.size(0), -1))
                batch_topk_idxes = torch.from_numpy(batch_topk_idxes)  # (bs, topk)

                mix_topk_features = torch.zeros(batch_features.size())
                weights = args.weights
                for i in range(args.topK):
                    batch_topi_features = torch.zeros(
                        batch_features.size())  # init
                    batch_topi_idxes = batch_topk_idxes[:, i]
                    for j, idx in enumerate(batch_topi_idxes):
                        topi_features = label2features[idx]
                        random_idx = torch.randint(0, topi_features.size(0), (1,))
                        batch_topi_features[j] = topi_features[random_idx]
                    mix_topk_features += weights[i] * batch_topi_features
                batch_topk_features = mix_topk_features.to(args.device)
                #########################################

                netE.zero_grad()
                netD.zero_grad()
                # ---------- VAE Encoder ---------- #
                mean, log_var = netE(batch_topk_features, text_features)
                std = torch.exp(0.5 * log_var)
                noise = torch.randn(mean.shape).to(args.device)
                z = std * noise + mean
                # ---------- VAE Decoder ---------- #
                inputD = torch.cat((z, text_features), dim=1)
                recon_features = netD(inputD)
                recon_features = recon_features / \
                    recon_features.norm(dim=-1, keepdim=True)

                # loss computation
                contra_loss = contrastive_loss(recon_features, text_features, temperature=args.t)
                reco_loss = recon_loss(
                    recon_features, batch_features, mean, log_var)
                loss = contra_loss * args.alpha1 + reco_loss * args.alpha2

                loss_list.append(loss.item())

                loss.backward()
                optimizerE.step()
                optimizerD.step()
            print(' ====> Train Loss: {:.4f}'.format(
                sum(loss_list)/len(loss_list)))
        # -------------------- Test Stage -------------------- #
        """
            - train loop epoches, then test
        """
        # ---------- synthesize unseen classes features ---------- #
        syn_unseen_f = []
        syn_unseen_l = []
        with torch.no_grad():
            netD.eval()
            in_package = {"topk_l": unseentopk, "l2f": label2features,
                            "VAE_E": netE, "sl2I": seenlabel2Idx}
            syn_unseen_f, syn_unseen_l = generate_syn_data(
                netD, myloader.unseenclasses, unseen_embeddings, in_package, args)
        syn_unseen_f = torch.from_numpy(
            syn_unseen_f).float().to(args.device)
        syn_unseen_l = torch.from_numpy(
            syn_unseen_l).squeeze(1).long().to(args.device)
        print(" ==> feature synthesized.")

        # ---------- train czsl classifier ---------- #
        new_train_f = syn_unseen_f
        new_train_l = syn_unseen_l

        _, new_train_l, _ = torch.unique(new_train_l, return_inverse=True, return_counts=True)
        new_trainset = TensorDataset(new_train_f, new_train_l)
        new_trainloader = DataLoader(new_trainset, batch_size=args.batch_size, shuffle=False)
        record_miniEpoch = 0.0

        for clf_epoch in range(args.clf_nepoch):
            loss_clf = []
            for batch_features, batch_labels in new_trainloader:
                netCLF.train()
                netCLF.zero_grad()
                batch_features, batch_labels = batch_features.to(
                    args.device), batch_labels.long().to(args.device)
                output = netCLF(batch_features)
                loss = criterion(output, batch_labels)
                loss_clf.append(loss.item())
                loss.backward()
                optimizerCLF.step()

            testloader = {"test_unseen_f": testUnseen_f, "test_unseen_l": testUnseen_l}
            acc_train = eval_train_acc(myloader, new_train_f, new_train_l, netCLF, args.device)
            CZSL = eval_zs_czsl(myloader, testloader, netCLF, args.device)
            print(f" ====> clf_epoch:{clf_epoch}; Loss:{sum(loss_clf)/len(loss_clf)}; Acc_train: {acc_train}")
            print(f" ====> CZSL:{CZSL}")

            if CZSL > record_miniEpoch:
                record_miniEpoch = CZSL

        early_break += 1
        if record_miniEpoch > best_CZSL:
            best_Epoch = epoch
            best_CZSL = record_miniEpoch
            # if args.save_model:
            #     torch.save(netE.state_dict(), f"{outlogPath}/{args.dataset}_netE_bestCZSL.pth")
            #     torch.save(netD.state_dict(), f"{outlogPath}/{args.dataset}_netD_bestCZSL.pth")
            #     torch.save(netCLF.state_dict(), f"{outlogPath}/{args.dataset}_netCLF_bestCZSL.pth")
            #     print('BestGZSL models saved!')
            early_break = 0

        dictNow = {'CZSL': record_miniEpoch}

        print("Performance => Epoch:{}; CZSL:{:.6f}".format(epoch, dictNow['CZSL']))
        print('Best GZSL => Epoch:{}; CZSL:{:.6f}'.format(best_Epoch, best_CZSL))
        print("Early break: ", early_break)

        logger.info("Performance => Epoch:{}; CZSL:{:.6f}".format(epoch, dictNow['CZSL']))
        logger.info('Best CZSL => Epoch:{}; CZSL:{:.6f}'.format(best_Epoch, best_CZSL))
        logger.info("-"*50)

        if early_break >= args.early_stop:
            print("Ealry break!")
            break


if __name__ == '__main__':
    args = parse_args()
    print('args:', args)
    main(args)
