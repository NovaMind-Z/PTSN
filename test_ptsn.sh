# test siwntransformer_base224_2k-800
python test_ptsn.py \
        --d_in 1024 \
        --IMG_SIZE 224 \
        --TESTCROP True \
        --img_root_path  '/path/to/data/coco_caption/IMAGE_COCO' \
        --model_path '/path/to/data/saved_models/swintransformer_base_texthiproto2000-800_best.pth' \
        --annotation_folder '/path/to/data/coco_caption/annotations' \
        --backbone_name 'swin_base_patch4_window7_224_22k'

## test siwntransformer_large384_2k-800
#python test_ptsn.py \
#        --d_in 1536 \
#        --IMG_SIZE 384 \
#        --TESTCROP False \
#        --img_root_path  '/path/to/data/coco_caption/IMAGE_COCO' \
#        --model_path '/path/to/data/saved_models/swintransformer_large384_texthiproto2k-800_best.pth' \
#        --annotation_folder '/path/to/data/coco_caption/annotations' \
#        --backbone_name 'swin_large_patch4_window12_384_22k'