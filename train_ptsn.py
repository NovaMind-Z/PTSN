import random
from data import TextField, RawField, ImageField
from data import COCO, DataLoader
from torch.utils.data import DistributedSampler
import evaluation
from evaluation import PTBTokenizer, Cider

from models.transformer import Transformer, TransformerDecoderLayer
from models.transformer.encoder_entry import build_encoder
from models.transformer.optimi_entry  import build_optimizer
from models.transformer.conceptencoders import PAEncoder

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import pickle
import numpy as np
import itertools
from shutil import copyfile
import torch.multiprocessing as mp
import torch.distributed as dist

import warnings

warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"

def evaluate_loss(model, dataloader, loss_fn, text_field, e, device):

    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader), disable=device!=0) as pbar:
        with torch.no_grad():
            for it, (detections, captions) in enumerate(dataloader):
                detections, captions = detections.to(device), captions.to(device)
                out = model(mode='xe', images=detections, seq=captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))

                this_loss = loss.item()

                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))


                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field, e, device):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader), disable=device!=0) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model(mode='rl', images=images, max_len=20, eos_idx=text_field.vocab.stoi['<eos>'], beam_size=5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, text_field, scheduler, loss_fn, e, device):
    # Training with cross-entropy
    model.train()
    scheduler.step()
    if device == 0:
        print('Backbone lr = ', optim.state_dict()['param_groups'][0]['lr'])
        print('Dec lr = ', optim.state_dict()['param_groups'][1]['lr'])
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader), disable=device!=0) as pbar:
        for it, (detections, captions) in enumerate(dataloader):
            detections, captions = detections.to(device), captions.to(device)
            # attrs = captions_attrs[:, :, 1]
            out = model(mode='xe', images=detections, seq=captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            loss.backward()
            optim.step()

            this_loss = loss.item()


            running_loss += this_loss


            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

            # scheduler.step()

    loss = running_loss / len(dataloader)

    return loss


def train_scst(model, dataloader, optim_rl, cider, text_field, scheduler_rl, e, device):
    # Training with self-critical
    # tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0

    model.train()
    scheduler_rl.step()
    if device == 0:
        print('lr = ', optim_rl.state_dict()['param_groups'][0]['lr'])

    running_loss = .0
    seq_len = 20
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader), disable=device!=0) as pbar:
        for it, (detections, caps_gt) in enumerate(dataloader):
            detections = detections.to(device)
            outs, log_probs = model(mode='rl', images=detections, max_len=seq_len, eos_idx=text_field.vocab.stoi['<eos>'], beam_size=beam_size, out_size=beam_size)
            optim_rl.zero_grad()  #log_probs [10, 5, 20]

            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen = evaluation.PTBTokenizer.tokenize(caps_gen)
            caps_gt = evaluation.PTBTokenizer.tokenize(caps_gt)
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size) # [10, 5]
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim_rl.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline




def _changeConfig(config, worldSize):
    config.xe_base_lr = config.xe_base_lr * worldSize * 1.0
    config.rl_base_lr = config.rl_base_lr * worldSize * 1.0


def _generalConfig(rank: int, worldSize: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "61888"
    torch.autograd.set_detect_anomaly(False)
    torch.backends.cudnn.benchmark = True
    random.seed(1234)
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", world_size=worldSize, rank=rank)


def train(rank, worldSize, args):
    _generalConfig(rank, worldSize)

    print('Rank{}: Transformer Training'.format(rank))
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))
    # Pipeline for image regions
    image_field = ImageField(config=args)
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)


    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits

    if not os.path.isfile('vocab.pkl'):
        print("Rank{}: Building vocabulary".format(rank))
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab.pkl', 'wb'))
    else:
        print('Rank{}: Loading from vocabulary'.format(rank))
        text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

    # DDP Model and dataloaders
    backbone = build_encoder(args)
    encoder = PAEncoder(d_in=args.d_in, d_model=512)
    decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'], d_model=512)
    torch.cuda.set_device(rank)
    model = Transformer(text_field.vocab.stoi['<bos>'], backbone, decoder, encoder)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model.to(rank), device_ids=[rank], output_device=rank, broadcast_buffers=False, find_unused_parameters=True)


    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    # ref_caps_train = list(train_dataset.text)
    ref_caps_train = train_dataset.text()
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})


    '''
    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)
    '''

    def lambda_lr(s):
        rank = dist.get_rank()
        if rank == 0:
            print("s:", s)
        if s <= 3:
            lr = args.xe_base_lr * s / 4
        elif s <= 10:
            lr = args.xe_base_lr
        elif s <= 12:
            lr = args.xe_base_lr * 0.2
        else:
            lr = args.xe_base_lr * 0.2 * 0.2
        return lr
    
    def lambda_lr_rl(s):
        refine_epoch = args.refine_epoch_rl
        rank = dist.get_rank()
        if rank == 0:
            print("rl_s:", s)
        if s <= refine_epoch:
            lr = args.rl_base_lr
        elif s <= refine_epoch + 3:
            lr = args.rl_base_lr * 0.2
        elif s <= refine_epoch + 6:
            lr = args.rl_base_lr * 0.2 * 0.2
        else:
            lr = args.rl_base_lr * 0.2 * 0.2 * 0.2
        return lr


    # Initial conditions
    optim = build_optimizer(model)
    scheduler = LambdaLR(optim, lambda_lr)

    optim_rl = build_optimizer(model)
    scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)


    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    use_rl = False
    best_cider = .0
    best_test_cider = 0.
    patience = 0
    start_epoch = 0

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = 'saved_transformer_models/%s_last.pth' % args.exp_name
        else:
            fname = 'saved_transformer_models/%s_best.pth' % args.exp_name

        if os.path.exists(fname):
            data = torch.load(fname, map_location={"cuda:0": f"cuda:{rank}"})
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=True)
            """
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            """
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            best_test_cider = data['best_test_cider']
            patience = data['patience']
            use_rl = data['use_rl']

            if use_rl: # fix not
                optim.load_state_dict(data['optimizer'])
                scheduler.load_state_dict(data['scheduler'])
            else:
                optim_rl.load_state_dict(data['optimizer'])
                scheduler_rl.load_state_dict(data['scheduler']) #bug here, please set last_epoch manually

            print('Resuming from epoch %d, validation loss %f, best cider %f, and best_test_cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider'], data['best_test_cider']))
            print('patience:', data['patience'])

    print("Rank{}: Training starts".format(rank))
    for e in range(start_epoch, start_epoch + 100):
        trainSampler = DistributedSampler(train_dataset, worldSize, rank)
        trainSampler.set_epoch(e)
        dataloader_train = DataLoader(train_dataset, sampler=trainSampler, batch_size=args.batch_size, pin_memory=True, drop_last=False, num_workers=args.workers, persistent_workers=True)

        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        dict_trainSampler = DistributedSampler(dict_dataset_train, worldSize, rank)
        dict_trainSampler.set_epoch(e)
        dict_dataloader_train = DataLoader(dict_dataset_train, sampler=dict_trainSampler, batch_size=args.batch_size // 5,  pin_memory=True, drop_last=False, num_workers=args.workers, persistent_workers=True)

        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)
        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, text_field, scheduler, loss_fn, e, rank)
            if rank == 0:
                writer.add_scalar('data/train_loss', train_loss, e)

        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim_rl, cider_train, text_field, scheduler_rl, e, rank)
            if rank == 0:
                writer.add_scalar('data/train_loss', train_loss, e)
                writer.add_scalar('data/reward', reward, e)
                writer.add_scalar('data/reward_baseline', reward_baseline, e)

        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field, e, rank)
        if rank == 0:
            writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field, e, rank)
        # print("Validation scores", scores)
        val_cider = scores['CIDEr']
        if rank == 0:
            print("Validation scores", scores)
            writer.add_scalar('data/val_cider', val_cider, e)
            writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
            writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
            writer.add_scalar('data/val_meteor', scores['METEOR'], e)
            writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, text_field, e, rank)
        # print("Test scores", scores)
        test_cider = scores['CIDEr']
        if rank == 0:
            print("Test scores", scores)
            writer.add_scalar('data/test_cider', test_cider, e)
            writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
            writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
            writer.add_scalar('data/test_meteor', scores['METEOR'], e)
            writer.add_scalar('data/test_rouge', scores['ROUGE'], e)

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        best_test = False
        if test_cider >= best_test_cider:
            best_test_cider = test_cider
            best_test = True

        switch_to_rl = False
        exit_train = False

        if patience == 5:
            if e < args.xe_least:   # xe stage train 15 epoches at least
                if rank == 0:
                    print('special treatment, e = {}'.format(e))
                use_rl = False
                switch_to_rl = False
                patience = 0
            elif not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                
                optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
                scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)
                
                for k in range(e-1):
                    scheduler_rl.step()
                if rank == 0:
                    print("Switching to RL")
            else:
                if rank == 0:
                    print('patience reached.')
                exit_train = True

        if e == args.xe_most:     # xe stage no more than 20 epoches
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                
                optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
                scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)

                for k in range(e-1):
                    scheduler_rl.step()
                print("Switching to RL")

        if rank == 0:
            if switch_to_rl and not best:
                data = torch.load('saved_transformer_models/%s_best.pth' % args.exp_name)
                torch.set_rng_state(data['torch_rng_state'])
                torch.cuda.set_rng_state(data['cuda_rng_state'])
                np.random.set_state(data['numpy_rng_state'])
                random.setstate(data['random_rng_state'])
                model.load_state_dict(data['state_dict'])
                print('Resuming from epoch %d, validation loss %f, best_cider %f, and best test_cider %f' % (
                    data['epoch'], data['val_loss'], data['best_cider'], data['best_test_cider']))

            torch.save({
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate(),
                'epoch': e,
                'val_loss': val_loss,
                'val_cider': val_cider,
                'state_dict': model.state_dict(),
                'optimizer': optim.state_dict() if not use_rl else optim_rl.state_dict(),
                'scheduler': scheduler.state_dict() if not use_rl else scheduler_rl.state_dict(),
                'patience': patience,
                'best_cider': best_cider,
                'best_test_cider': best_test_cider,
                'use_rl': use_rl,
            }, 'saved_transformer_models/%s_last.pth' % args.exp_name)

            if best:
                copyfile('saved_transformer_models/%s_last.pth' % args.exp_name, 'saved_transformer_models/%s_best.pth' % args.exp_name)
            if best_test:
                copyfile('saved_transformer_models/%s_last.pth' % args.exp_name, 'saved_transformer_models/%s_best_test.pth' % args.exp_name)

            # save model
            if e >= 25:
                copyfile('saved_transformer_models/%s_last.pth' % args.exp_name, 'saved_transformer_models/{}_{}.pth'.format(args.exp_name, e))

        if exit_train:
            if rank == 0:
                writer.close()
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Progressive Tree-Structured prototype Network')
    parser.add_argument('--exp_name', type=str, default='swintransformer_base_texthiproto2000-800')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--max_detections', type=int, default=49)
    parser.add_argument('--resume_last', type=bool, default=False)
    parser.add_argument('--resume_best', action='store_true')

    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    parser.add_argument('--xe_least', type=int, default=15)
    parser.add_argument('--xe_most', type=int, default=20)
    parser.add_argument('--refine_epoch_rl', type=int, default=28)
    parser.add_argument('--xe_base_lr', type=float, default=0.0001)
    parser.add_argument('--rl_base_lr', type=float, default=5e-6)


    ### data aug config
    parser.add_argument('--IMG_SIZE', type=int, default=224)
    parser.add_argument('--COLOR_JITTER', type=float, default=0.4)
    parser.add_argument('--AUTO_AUGMENT', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--REPROB', type=float, default=0.25)
    parser.add_argument('--REMODE', type=str, default='pixel')
    parser.add_argument('--RECOUNT', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--d_in', type=int, default=1024)
    parser.add_argument('--INTERPOLATION', type=str, default='bicubic')
    parser.add_argument('--TESTCROP', type=bool, default=True)
    parser.add_argument('--img_root_path', type=str, default='~/data/dataset/coco_caption/IMAGE_COCO')
    parser.add_argument('--annotation_folder', type=str, default='~/data/dataset/coco_caption/annotations')
    parser.add_argument('--backbone_resume_path', type=str, default='~/data/resume_model/swin_base_patch4_window7_224_22k.pth')
    parser.add_argument('--backbone_name', type=str, default='swin_base_patch4_window7_224_22k')

    args = parser.parse_args()
    print(args)


    ## DDP Training
    worldSize = args.num_gpus
    _changeConfig(args, worldSize)
    print('\nDistribute config', args)
    mp.spawn(train, (worldSize, args), worldSize)