export CUDA_VISIBLE_DEVICES=0

python main_image_test2.py \
    --batch_size 128 \
    --cls_token \
    --finetune pretrained/mae_pretrain_vit_b_edit.pth \
    --data_path PBVS/test \
    --resume pretrained/checkpoint-best.pth \
    --drop_path 0.0 \
    --blr 0.1 \
    --dataset cifar100 \
    --nb_classes 10 \
    --ffn_adapt \
    --output_dir test \
    --epochs 50 \
    --eval