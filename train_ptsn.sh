python train_ptsn.py \
        --IMG_SIZE 224 \
        --img_root_path '/path/to/data/coco_caption/IMAGE_COCO' \
        --backbone_resume_path '/path/to/data/resume_model/swin_base_patch4_window7_224_22k.pth' \
        --annotation_folder '/path/to/data/coco_caption/annotations/' \
        --num_gpus 4

