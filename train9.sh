# image

export CUDA_VISIBLE_DEVICES=0

python main_image_seventh.py \
    --batch_size 128 \
    --cls_token \
    --finetune pretrained/mae_pretrain_vit_b_edit.pth \
    --data_path Unicorn_Dataset/train/ \
    --drop_path 0.0 \
    --blr 0.1 \
    --dataset cifar100 \
    --nb_classes 10 \
    --ffn_adapt \
    --output_dir experiments/exp09 \
    --epochs 50