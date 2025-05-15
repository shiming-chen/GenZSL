import os
import ast
import argparse


def parse_args():
    rootPath = os.path.dirname(__file__)
    datarootPath = rootPath + "/dataset"
    parser = argparse.ArgumentParser(description="")
    # -------------------- Path config --------------------#
    parser.add_argument('--projectPath', default=rootPath, help='')
    parser.add_argument('--dataset', default='CUB', help='dataset: AWA2/CUB/SUN')
    parser.add_argument('--image_root', default= datarootPath + '/data/dataset', help='Path to image root')
    parser.add_argument('--mat_path', default= datarootPath + '/data/dataset/xlsa17/data',
                        help='Features extracted from pre-training Resnet')
    parser.add_argument('--log_root_path', default=rootPath + '/out', help='Save path of exps')
    # -------------------- train config --------------------#
    parser.add_argument('--backbone', default='ViT-B/16', help='CLIP backbone')
    parser.add_argument('--inputsize', type=int, default=224, help='input_imgage_size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for classifier')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size')
    parser.add_argument('--nepoch', type=int, default=150, help='Number of epoches to train')
    parser.add_argument('--clf_nepoch', type=int, default=20, help='Number of epoches to train classifier')
    parser.add_argument('--syn_num', type=int, default=800, help='Number of synthesized samples per class')
    parser.add_argument('--loop', type=int, default=3, help='Number of VAE pretraining')
    parser.add_argument('--alpha1', default=0.0, type=float, help='weight for contrastive loss')
    parser.add_argument('--alpha2', default=1.0, type=float, help='weight for reconsturction loss')
    # -------------------- other config --------------------#
    parser.add_argument('--seed', default=2024, type=int, help='seed for reproducibility')
    parser.add_argument('--t', default=0.07, type=float, help='temperature parameter')
    parser.add_argument('--early_stop', default=100, type=int, help='early stop')
    parser.add_argument('--topK', default=2, type=int, help='top k similarity')
    parser.add_argument('--weights', default=[0.8, 0.2], type=ast.literal_eval, help='top k similarity')
    parser.add_argument('--save_model', action='store_true', default=True, help='Save the trained model or not')
    parser.add_argument('--use_svd', action='store_true', default=False, help='whether to enable svd')
    parser.add_argument('--device', default='cuda:3', help='cpu/cuda:x')
    args = parser.parse_args()
    return args
