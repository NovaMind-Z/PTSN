import random
from data import TextField, RawField, ImageField
from data import COCO, DataLoader
import evaluation
from models.transformer import Transformer, TransformerDecoderLayer
from models.transformer.encoder_entry import build_encoder
from models.transformer.conceptencoders import PAEncoder

import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def predict_captions(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)

    return scores


if __name__ == '__main__':
    start_time = time.time()
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--m', type=int, default=40)

    parser.add_argument('--annotation_folder', type=str, default='~/data/dataset/coco_caption/annotations')

    # the path of tested model and vocabulary
    parser.add_argument('--model_path', type=str, default='~/data/saved_transformer_models/swintransformer_base_texthiproto2000-800_v3_best.pth')
    # parser.add_argument('--model_path', type=str, default='~/data/saved_transformer_models/swintransformer_large384_v2_best_test.pth')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl')

    ### data aug config
    parser.add_argument('--IMG_SIZE', type=int, default=224)
    parser.add_argument('--COLOR_JITTER', type=float, default=0.4)
    parser.add_argument('--AUTO_AUGMENT', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--REPROB', type=float, default=0.25)
    parser.add_argument('--REMODE', type=str, default='pixel')
    parser.add_argument('--RECOUNT', type=int, default=1)
    parser.add_argument('--INTERPOLATION', type=str, default='bicubic')
    parser.add_argument('--TESTCROP', type=bool, default=True)
    parser.add_argument('--img_root_path', type=str, default='~/data/dataset/coco_caption/IMAGE_COCO')
    parser.add_argument('--backbone_resume_path', type=str, default=None)
    parser.add_argument('--backbone_name', type=str, default=None)
    parser.add_argument('--d_in', type=int, default=1024)

    args = parser.parse_args()

    print('Transformer Evaluation')

    # Pipeline for image regions
    image_field = ImageField(config=args)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    _, _, test_dataset = dataset.splits
    text_field.vocab = pickle.load(open(args.vocab_path, 'rb'))

    # Model and dataloaders
    backbone = build_encoder(args)
    encoder = PAEncoder(d_in=args.d_in, d_model=512)
    decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'], d_model=512)
    model = Transformer(text_field.vocab.stoi['<bos>'], backbone, decoder, encoder).to(device)

    data = torch.load(args.model_path)
    state_dict = {}
    for k, v in data['state_dict'].items():
        state_dict[k.split('module.')[-1]] = v
    model.load_state_dict(state_dict)

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    scores = predict_captions(model, dict_dataloader_test, text_field)
    print(scores)
    print('it costs {} s to test.'.format(time.time() - start_time))