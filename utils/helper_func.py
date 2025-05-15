
import clip
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F


def contrastive_loss(image_features, text_features, temperature=0.07):
    logits = torch.matmul(image_features, text_features.t()) / temperature
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2


def recon_loss(recon_f, origin_f, mean, log_var):
    """
        - Reconstruction Loss
    """
    REC = (recon_f - origin_f).pow(2).sum(1).mean()
    KLD = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum(dim=1).mean()
    return (REC + 1 * KLD)


def eval_zs_gzsl(dataloader, testloader, model, device):
    model.eval()
    test_seen_feature = testloader["test_seen_f"]
    test_seen_label = testloader["test_seen_l"].to(device)

    test_unseen_feature = testloader["test_unseen_f"]
    test_unseen_label = testloader["test_unseen_l"].to(device)
    seenclasses = dataloader.seenclasses
    unseenclasses = dataloader.unseenclasses

    batch_size = 100
    in_package = {'model': model, 'device': device, 'batch_size': batch_size}

    with torch.no_grad():
        acc_seen = val_gzsl(test_seen_feature, test_seen_label, seenclasses, in_package)
        acc_novel, acc_zs = val_zs_gzsl(test_unseen_feature, test_unseen_label, unseenclasses, in_package)
    if (acc_seen+acc_novel) > 0:
        H = (2*acc_seen*acc_novel) / (acc_seen+acc_novel)
    else:
        H = 0
    return acc_seen, acc_novel, H, acc_zs


def eval_zs_czsl(dataloader, testloader, model, device):
    model.eval()

    test_unseen_feature = testloader["test_unseen_f"]
    test_unseen_label = testloader["test_unseen_l"].to(device)
    unseenclasses = dataloader.unseenclasses
    batch_size = 100

    with torch.no_grad():
        start = 0
        ntest = test_unseen_feature.size()[0]
        predicted_label = torch.LongTensor(test_unseen_label.size()).to(device)

        for _ in range(0, ntest, batch_size):
            end = min(ntest, start+batch_size)
            input = test_unseen_feature[start:end].to(device)
            output= model(input)
            predicted_label[start:end] = torch.argmax(output.data, 1)
            start = end
        acc_czsl = (predicted_label == test_unseen_label).sum().float() / len(test_unseen_label)
    return acc_czsl


def val_gzsl(test_X, test_label, target_classes, in_package):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]  # number of samples
        predicted_label = torch.LongTensor(test_label.size())
        for _ in range(0, ntest, batch_size):
            end = min(ntest, start+batch_size)
            input = test_X[start:end].to(device)
            output= model(input)
            predicted_label[start:end] = torch.argmax(output.data, 1)
            start = end
        acc = compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, in_package)
        return acc


def val_zs_gzsl(test_X, test_label, unseen_classes, in_package):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        predicted_label_gzsl = torch.LongTensor(test_label.size())
        predicted_label_zsl = torch.LongTensor(test_label.size())
        predicted_label_zsl_t = torch.LongTensor(test_label.size())
        for i in range(0, ntest, batch_size):
            end = min(ntest, start+batch_size)
            input = test_X[start:end].to(device)
            output = model(input)
            output_t = output.clone()
            output_t[:, unseen_classes] = output_t[:,unseen_classes]+torch.max(output)+1
            predicted_label_zsl[start:end] = torch.argmax(output_t.data, 1)
            predicted_label_zsl_t[start:end] = torch.argmax(
                output.data[:, unseen_classes], 1)
            predicted_label_gzsl[start:end] = torch.argmax(output.data, 1)
            start = end
        acc_gzsl = compute_per_class_acc_gzsl(
            test_label, predicted_label_gzsl, unseen_classes, in_package)
        # acc_zs = compute_per_class_acc_gzsl(test_label, predicted_label_zsl, unseen_classes, in_package)
        acc_zs_t = compute_per_class_acc(map_label(
            test_label, unseen_classes), predicted_label_zsl_t, unseen_classes.size(0))
        # assert np.abs(acc_zs - acc_zs_t) < 0.001
        # print('acc_zs: {} acc_zs_t: {}'.format(acc_zs,acc_zs_t))
        return acc_gzsl, acc_zs_t


def compute_per_class_acc(test_label, predicted_label, nclass):
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    for i in range(nclass):
        idx = (test_label == i)
        acc_per_class[i] = torch.sum(
            test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
    return acc_per_class.mean().item()


def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, in_package):
    device = in_package['device']
    per_class_accuracies = torch.zeros(
        target_classes.size()[0]).float().to(device).detach()
    predicted_label = predicted_label.to(device)
    for i in range(target_classes.size()[0]):
        is_class = test_label == target_classes[i]
        per_class_accuracies[i] = torch.div(
            (predicted_label[is_class] == test_label[is_class]).sum().float(), is_class.sum().float())
    return per_class_accuracies.mean().item()


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size()).fill_(-1)
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label


def eval_train_acc(dataloader, test_f, test_l, model, device):
    model.eval()
    test_feature = test_f
    test_label = test_l.to(device)
    batch_size = 100
    with torch.no_grad():
        start = 0
        ntest = test_feature.size()[0]  # number of samples
        predicted_label = torch.LongTensor(test_label.size()).to(device)
        for _ in range(0, ntest, batch_size):
            end = min(ntest, start+batch_size)
            input = test_feature[start:end].to(device)
            output= model(input)
            predicted_label[start:end] = torch.argmax(output.data, 1)
            start = end
        acc = (predicted_label == test_label).sum().float() / len(test_label)
        return acc