import torch.nn as nn
from controllers.encoder_controller import encoder

class Seq2Seq_model(nn.Module):
    def __init__(self, attributes, config):

        super(Seq2Seq_model, self).__init__()

        self.config              = config
        self.UNK_id              = attributes['unk_id']
        self.config['vocab_len'] = attributes['vocab_len']
        self.config['pad_id']    = attributes['pad_id']
        self.config['sep_id']    = attributes['sep_id']
        self.config['unk_id']    = attributes['unk_id']
        self.encoder_decoder     = encoder(self.config)

    def forward(self, batch):
        
        if 'trg_vec' in batch:
            trg, trg_mask = batch['trg_vec'], batch['trg_mask']
        else:
            trg, trg_mask = None, None
    
        src           = batch['src_vec']
        src_mask      = batch['src_mask']
        sequence_dict = self.encoder_decoder(src_idx=src, input_mask=src_mask, trg_idx=trg, output_mask=trg_mask)

        return sequence_dict
