import torch
from models.transformer.swintransformer import SwinTransformer


def build_encoder(config):
    backbone_name = config.backbone_name
    if backbone_name == 'swin_base_patch4_window7_224_22k':
        model = SwinTransformer(img_size=224,
                        patch_size=4,
                        in_chans=3,
                        num_classes=21841,
                        embed_dim=128,
                        depths=[ 2, 2, 18, 2 ],
                        num_heads=[ 4, 8, 16, 32 ],
                        window_size=7,
                        mlp_ratio=4.,
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0.0,
                        drop_path_rate=0.5,
                        ape=False,
                        patch_norm=True,
                        use_checkpoint=False)
    elif backbone_name == 'swin_large_patch4_window12_384_22k':
        model = SwinTransformer(img_size=384,
                        patch_size=4,
                        in_chans=3,
                        num_classes=21841,
                        embed_dim=192,
                        depths=[2, 2, 18, 2],
                        num_heads=[6, 12, 24, 48],
                        window_size=12,
                        mlp_ratio=4.,
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0.0,
                        ape=False,
                        patch_norm=True,
                        use_checkpoint=False)
    else:
        print('The given model does not exist!')
    backbone_resume_path = config.backbone_resume_path
    if not backbone_resume_path is None:
        checkpoint = torch.load(backbone_resume_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=True)
        print(msg)
    else:
        print('resume from random init weights')


    return model


#swin_large_patch4_window12_384_22k

# def build_encoder(config):
#     model = SwinTransformer(img_size=384,
#                             patch_size=4,
#                             in_chans=3,
#                             num_classes=21841,
#                             embed_dim=192,
#                             depths=[ 2, 2, 18, 2 ],
#                             num_heads=[ 6, 12, 24, 48 ],
#                             window_size=12,
#                             mlp_ratio=4.,
#                             qkv_bias=True,
#                             qk_scale=None,
#                             drop_rate=0.0,
#                             ape=False,
#                             patch_norm=True,
#                             use_checkpoint=False)
#     swin_resume_path = config.swin_resume_path
#
#     if not swin_resume_path is None:
#         checkpoint = torch.load(swin_resume_path, map_location='cpu')
#         msg = model.load_state_dict(checkpoint['model'], strict=True)
#         print(msg)
#     else:
#         print('resume from random init weights')
#
#
#     return model

#swin_base_patch4_window12_384_22k
# def build_encoder(config):
#     model = SwinTransformer(img_size=384,
#                             patch_size=4,
#                             in_chans=3,
#                             num_classes=21841,
#                             embed_dim=128,
#                             depths=[ 2, 2, 18, 2 ],
#                             num_heads=[ 4, 8, 16, 32 ],
#                             window_size=12,
#                             mlp_ratio=4.,
#                             qkv_bias=True,
#                             qk_scale=None,
#                             drop_rate=0.0,
#                             ape=False,
#                             patch_norm=True,
#                             use_checkpoint=False)
#     swin_resume_path = config.swin_resume_path
#
#     checkpoint = torch.load(swin_resume_path, map_location='cpu')
#     msg = model.load_state_dict(checkpoint['model'], strict=True)
#     print(msg)
#
#
#     return model