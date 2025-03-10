export CUDA_VISIBLE_DEVICES=0

python main_image_test2.py \
    --batch_size 128 \
    --cls_token \
    --finetune /home/mjh/challenge/pbvs2025/AdaptFormer/pretrained/mae_pretrain_vit_b_edit.pth \
    --data_path /home/whisper2024/PBVS/test \
    --resume /home/mjh/challenge/pbvs2025/AdaptFormer/pretrained/checkpoint-best.pth \
    --drop_path 0.0 \
    --blr 0.1 \
    --dataset cifar100 \
    --nb_classes 10 \
    --ffn_adapt \
    --output_dir test/exp09/epoch20 \
    --epochs 50 \
    --eval


#  python main_image_test2.py \
#         --batch_size 128 \
#         --cls_token \
#         --finetune /workspace/hjkim/AdaptFormer/mae_pretrain_vit_b_edit.pth \
#         --data_path /workspace/hjkim/PBVS_dataset/Unicorn_Dataset/test \
#         --resume /workspace/hjkim/AdaptFormer/experiments/exp09/checkpoint-23.pth \
#         --drop_path 0.0 \
#         --blr 0.1 \
#         --dataset cifar100 \
#         --nb_classes 10 \
#         --ffn_adapt \
#         --output_dir test/exp09/epoch23 \
#         --epochs 50 \
#         --eval

#  python main_image_test2.py \
#         --batch_size 128 \
#         --cls_token \
#         --finetune /workspace/hjkim/AdaptFormer/mae_pretrain_vit_b_edit.pth \
#         --data_path /workspace/hjkim/PBVS_dataset/Unicorn_Dataset/test \
#         --resume /workspace/hjkim/AdaptFormer/experiments/exp09/checkpoint-25.pth \
#         --drop_path 0.0 \
#         --blr 0.1 \
#         --dataset cifar100 \
#         --nb_classes 10 \
#         --ffn_adapt \
#         --output_dir test/exp09/epoch25 \
#         --epochs 50 \
#         --eval