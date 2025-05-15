# for the first time
python extract_CLIP_feature.py --dataset AWA2
python extract_CLIP_feature.py --dataset CUB
python extract_CLIP_feature.py --dataset SUN

# Once have saved CLIP features
# ======================================== AWA2 ======================================== #
python train_topk_f_gzsl.py --dataset AWA2 --syn_num 5000 --alpha1 0.1 --use_svd --topK 2
python train_topk_f_czsl.py --dataset AWA2 --syn_num 5000 --alpha1 0.1 --use_svd --topK 2
# ======================================== CUB ======================================== #
python train_topk_f_gzsl.py --dataset CUB --syn_num 1600 --alpha1 0.1 --use_svd --topK 2
python train_topk_f_czsl.py --dataset CUB --syn_num 1600 --alpha1 0.1 --use_svd --topK 2
# ======================================== SUN ======================================== #
python train_topk_f_gzsl.py --dataset SUN --syn_num 3200 --alpha1 0.001 --use_svd --topK 2
python train_topk_f_czsl.py --dataset SUN --syn_num 3200 --alpha1 0.001 --use_svd --topK 2


