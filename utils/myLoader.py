"""
    - load data from dataset
"""
import re
import torch
import numpy as np
import scipy.io as sio


class MyDataloader():
    def __init__(self, args):
        """
            - args: need
                mat_path -> xlsa17 root path
                dataset  -> dataset_name ([AWA,CUB,SUN,FLO])
                image_root -> local dataset root path
        """
        # get labels & image_names
        res101 = sio.loadmat(args.mat_path + f"/{args.dataset}/res101.mat")
        self.label = res101['labels'].astype(int).squeeze() - 1
        self.image_files = res101['image_files'].squeeze()
        # get sample idxs in mat
        att_splits = sio.loadmat(args.mat_path + f"/{args.dataset}/att_splits.mat")

        self.trainval_loc = att_splits['trainval_loc'].squeeze() - 1
        # self.train_loc = att_splits['train_loc'].squeeze() - 1
        # self.val_unseen_loc = att_splits['val_loc'].squeeze() - 1
        self.test_seen_loc = att_splits['test_seen_loc'].squeeze() - 1
        self.test_unseen_loc = att_splits['test_unseen_loc'].squeeze() - 1
        # get label idxs in mat
        self.test_seen_label = torch.from_numpy(self.label[self.test_seen_loc]).long()
        self.test_unseen_label = torch.from_numpy(self.label[self.test_unseen_loc]).long()
        self.seenclasses = torch.from_numpy(np.unique(self.test_seen_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        # get image_file_dataframe
        self.train_df = convert_path(args, self.image_files, self.trainval_loc, self.label)
        self.test_seen_df = convert_path(args, self.image_files, self.test_seen_loc, self.label)
        self.test_unseen_df = convert_path(args, self.image_files, self.test_unseen_loc, self.label)
        if args.dataset in ["AWA2", "CUB", "SUN"]:
            # get class_names
            self.allclasses_name = att_splits['allclasses_names']
            # get seen/unseen class names
            self.seen_names = [self.allclasses_name[self.seenclasses][i][0][0] for i in range(len(self.seenclasses))]
            self.seen_names = [re.sub(r'\d+\.', '', item) for item in self.seen_names]
            self.unseen_names = [self.allclasses_name[self.unseenclasses][i][0][0] for i in range(len(self.unseenclasses))]
            self.unseen_names = [re.sub(r'\d+\.', '', item) for item in self.unseen_names]
            # get class_names to idxs mapping
            self.seenClass2Idx = {value: idx for idx, value in enumerate(self.seen_names)}
            self.unseenClass2Idx = {value: idx for idx, value in enumerate(self.unseen_names)}
        else:
            with open(args.mat_path + f"/{args.dataset}/class2label.txt", 'r') as f:
                lines = [line.strip() for line in f]
            self.allclasses_name = lines
            self.seen_names = [self.allclasses_name[seenclass] for seenclass in self.seenclasses]
            self.unseen_names = [self.allclasses_name[unseenclass] for unseenclass in self.unseenclasses]
            self.seenClass2Idx = {value: idx for idx, value in enumerate(self.seen_names)}
            self.unseenClass2Idx = {value: idx for idx, value in enumerate(self.unseen_names)}
    

def convert_path(args, image_files, img_loc, image_labels):
    """
        - convert mat path to local path
        - return a list of tuples -> (img_file, img_label)
    """
    imglist = []
    image_files = image_files[img_loc]
    image_labels = image_labels[img_loc]
    for image_file, image_label in zip(image_files, image_labels):
        if args.dataset == 'AWA2':
            image_file = args.image_root + '/AWA2/' + '/'.join(image_file[0].split('/')[5:])
        elif args.dataset == 'CUB':
            image_file = args.image_root + '/CUB/' + '/'.join(image_file[0].split('/')[6:])
        elif args.dataset == 'FLO':
            image_file = args.image_root + '/FLO/' + '/'.join(image_file.split('/')[-2:])
        elif args.dataset == 'SUN':
            image_file = args.image_root + '/SUN/' + '/'.join(image_file[0].split('/')[7:])
        else:
            raise ValueError("Unkonwn dataset!")
        imglist.append((image_file, int(image_label)))
    return imglist




