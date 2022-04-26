import torch
from torch import nn
import copy
from models.containers import ModuleList
from ..captioning_model import CaptioningModel
from models.beam_search import *


class Transformer(CaptioningModel):
    def __init__(self, bos_idx, backbone, decoder, encoder):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and not 'backbone' in name:
                if 'word_emb' in name or 'pos_emb' or 'decoder.fc' in name:
                    continue
                nn.init.xavier_uniform_(p)


    def forward(self, mode, images, seq=None, attrs=None, max_len=None, eos_idx=None, beam_size=None, out_size=1, return_probs=False):
        if mode == 'xe':
            features, mask_enc = self.backbone(images)
            enc_output = self.encoder(features)
            dec_output = self.decoder(seq, enc_output, mask_enc)
            return dec_output
        elif mode == 'rl':
            bs = BeamSearch(self, max_len, eos_idx, beam_size)
            return bs.apply(images, out_size, return_probs)

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                features, _ = self.backbone(visual)
                self.enc_output = self.encoder(features)
                self.mask_enc = (torch.sum(self.enc_output, -1) == 0).unsqueeze(1).unsqueeze(1)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc)


class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)
