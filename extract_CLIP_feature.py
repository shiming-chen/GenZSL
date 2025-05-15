import tqdm
import clip
import h5py
import torch
import argparse
import numpy as np
from PIL import Image
from utils.myLoader import MyDataloader

# -------------------- Config --------------------#
datarootPath = "/data/fudingjie/ZeroShot"
parser = argparse.ArgumentParser(description="")
parser.add_argument('--dataset', default='AWA2', help='dataset: AWA2/CUB/SUN')
parser.add_argument('--image_root', default= datarootPath + '/data/dataset', help='Path to image root')
parser.add_argument('--mat_path', default= datarootPath + '/data/dataset/xlsa17/data',
                    help='Features extracted from pre-training Resnet')
parser.add_argument('--backbone', default='ViT-B/16', help='CLIP backbone')
parser.add_argument('--seed', default=2024, type=int, help='seed for reproducibility')
parser.add_argument('--device', default='cuda:0', help='cpu/cuda:x')
args = parser.parse_args()


def get_textEmbedding(classnames, clip_model, device, norm=True):
    """
        - CLIP text embeddings
        - Note: features are normalized
    """
    with torch.no_grad():
        classnames = [classname.replace('_', ' ') for classname in classnames]
        if args.dataset == "AWA2":
            classnames = [classname.replace('+', ' ') for classname in classnames]
        text_descriptions = [f"A photo of a {classname}." for classname in classnames]
        text_tokens = clip.tokenize(text_descriptions, context_length=77).to(device)
        # prompt ensemble for ImageNet
        text_features = clip_model.encode_text(text_tokens).float().to(device)
        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)
        class_embeddings = text_features.to(device)
    return class_embeddings


def get_visualEmbedding(clip_model, dataframe, device, transform=None):
    """
        - CLIP visual embeddings
        - Note: features are normalized
    """
    with torch.no_grad():
        features = []
        labels = []
        progress = tqdm.tqdm(total=len(dataframe), ncols=100)
        for img_path, label in dataframe:
            progress.update(1)
            img = Image.open((img_path)).convert('RGB')
            if transform is not None:
                img = transform(img)
            img = img.unsqueeze(0).to(device)
            feature = clip_model.encode_image(img).float().to(device)
            feature /= feature.norm(dim=-1, keepdim=True)
            features.append(feature.cpu())
            labels.append(label)
        progress.close()
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    return features, labels


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# ======================================== CLIP ======================================== #
clip_model, preprocess = clip.load(args.backbone)
clip_model.to(args.device)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False
# ======================================== Prepare dataset ======================================== #
myloader = MyDataloader(args)
# maps
seenlabel2Idx = {value.item(): idx for idx, value in enumerate(myloader.seenclasses)}
seenidx2Label = {idx:value.item() for idx, value in enumerate(myloader.seenclasses)}
unseenlabel2Idx = {value.item(): idx for idx, value in enumerate(myloader.unseenclasses)}
unseenidx2Label = {idx:value.item() for idx, value in enumerate(myloader.unseenclasses)}
# class names
seen_names = myloader.seen_names # seen names
unseen_names = myloader.unseen_names # unseen names
# dataframes
train_dataframe = myloader.train_df
test_seen_dataframe = myloader.test_seen_df
test_unseen_dataframe = myloader.test_unseen_df
# ======================================== Textual features ======================================== #
print(" ==> Getting textual features from CLIP's text Encoder.")
seen_embeddings = get_textEmbedding(seen_names, clip_model, args.device).cpu()
unseen_embeddings = get_textEmbedding(unseen_names, clip_model, args.device).cpu()
# print(seen_embeddings.shape, unseen_embeddings.shape)
# ======================================== Visual features ======================================== #
print(" ==> Getting visual features from CLIP's visual Encoder.")
CLIP_feature_path = args.image_root + f"/{args.dataset}/CLIP_feature.hdf5"

print(" ====> Extract feature now.")
orig_f_train, labels_train = get_visualEmbedding(clip_model, train_dataframe, args.device, transform=preprocess)
print(" ========> train data finished.")
orig_f_testSeen, labels_testSeen = get_visualEmbedding(clip_model, test_seen_dataframe, args.device, transform=preprocess)
print(" ========> testSeen data finished.")
orig_f_testUnseen, labels_testUnseen = get_visualEmbedding(clip_model, test_unseen_dataframe, args.device, transform=preprocess)
print(" ========> testUnseen data finished.")

f = h5py.File(CLIP_feature_path, "w")
f.create_dataset('train_f', data = orig_f_train, compression = "gzip")
f.create_dataset('train_l', data = labels_train, compression = "gzip")
f.create_dataset('testSeen_f', data = orig_f_testSeen, compression = "gzip")
f.create_dataset('testSeen_l', data = labels_testSeen, compression = "gzip")
f.create_dataset('testUnseen_f', data = orig_f_testUnseen, compression = "gzip")
f.create_dataset('testUnseen_l', data = labels_testUnseen, compression = "gzip")
f.create_dataset('seen_embeddings', data = seen_embeddings, compression = "gzip")
f.create_dataset('unseen_embeddings', data = unseen_embeddings, compression = "gzip")
f.close()
print(" ====> Feature saved.")
