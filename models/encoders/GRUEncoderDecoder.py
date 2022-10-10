import numpy as np, torch as T, torch.nn as nn, torch.nn.functional as F, pdb
from models.layers import Linear
from models.attentions import attention

class GRUEncoderDecoder(nn.Module):
    def __init__(self, config):
        super(GRUEncoderDecoder, self).__init__()

        self.sos_token           = nn.Parameter(T.randn(config['embd_dim']))
        self.pad_inf             = -1.0e10
        self.vocab_len           = config['vocab_len']
        self.config              = config
        self.dropout             = config['dropout']
        self.UNK_id              = config['unk_id']
        self.embed_layer         = nn.Embedding(self.vocab_len, config['embd_dim'], padding_idx=config['pad_id'])
        self.encoder_hidden_size = config['encoder_hidden_size']
        self.decoder_hidden_size = config['decoder_hidden_size']
        self.encoder             = nn.GRU(input_size=config['embd_dim'], hidden_size=config['encoder_hidden_size'], num_layers=config['encoder_layers'], batch_first=True, bidirectional=True)
        self.decodercell         = nn.GRUCell(input_size=config['embd_dim'], hidden_size=config['decoder_hidden_size'], bias=True)
        self.attention           = attention(config)
        self.out_linear1         = Linear(2*self.encoder_hidden_size + self.decoder_hidden_size, self.decoder_hidden_size)
        self.out_linear2         = Linear(self.decoder_hidden_size, self.vocab_len)
        self.pointer_linear      = Linear(2*self.encoder_hidden_size + self.decoder_hidden_size + config['embd_dim'], 1)
        self.eps                 = 1e-9
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed_layer.weight.data.uniform_(-initrange, initrange)
        self.sos_token.data.uniform_(-initrange, initrange)

    def forward(self, src_idx, input_mask, trg_idx=None, output_mask=None):
        src           = self.embed_layer(src_idx)
        tfr           = self.config['teacher_force_ratio']
        teacher_force = np.random.choice([True, False], p=[tfr, 1 - tfr])

        if trg_idx is not None:
            assert output_mask is not None
            trg = self.embed_layer(trg_idx)
        else:
            trg = None

        N, S1, D = src.size()
        if trg is not None:
            N, S2, D = trg.size() 

        ptr_src_idx_if_unk = T.arange(start=self.vocab_len, end=self.vocab_len + S1).long().unsqueeze(0).repeat(N, 1).to(src.device)
        pointer_src_idx    = T.where(src_idx == self.UNK_id, ptr_src_idx_if_unk, src_idx)
        attention_mask     = T.where(input_mask == 0.0, T.empty_like(input_mask).fill_(self.pad_inf).float().to(input_mask.device), T.zeros_like(input_mask).float().to(input_mask.device))
        
        attention_mask               = attention_mask.unsqueeze(-1)
        assert attention_mask.size() == (N, S1, 1)
        
        lengths         = T.sum(input_mask, dim=1).long().view(N).cpu()
        packed_sequence = nn.utils.rnn.pack_padded_sequence(src, lengths, batch_first=True, enforce_sorted=False)
        encoded_src, hn = self.encoder(packed_sequence)
        encoded_src, _  = nn.utils.rnn.pad_packed_sequence(encoded_src, batch_first=True)
        
        assert encoded_src.size() == (N, S1, 2 * self.encoder_hidden_size)
        assert hn.size()          == (2, N, self.encoder_hidden_size)

        hn              = hn.permute(1, 0, 2).contiguous()
        assert hn.size() == (N, 2, self.encoder_hidden_size)
        hn              = hn.view(N, 2 * self.encoder_hidden_size)

        # pdb.set_trace()
        input = self.sos_token.view(1, D).repeat(N, 1)

        h = hn
        # S = S2 if not self.config['generate'] else self.config['max_decoder_len']
        if not self.config["generate"]:
            S = S2
        else:
            S = self.config["max_decoder_len"]

        key_encoded_src   = encoded_src.clone()
        value_encoded_src = encoded_src.clone()

        output_dists = []

        for i in range(S):
            if i > 0:
                if not self.config["generate"] and teacher_force:
                    input = trg[:, i - 1, :]
                else:
                    input = self.embed_layer(input_idx)

            h = self.decodercell(input, h)

            attention_scores, context_vector, key_encoded_src = self.attention(key_encoder_states=key_encoded_src, value_encoder_states=value_encoded_src, decoder_state=h, attention_mask=attention_mask, input_mask=input_mask)

            if not self.config['key_value_attention']:
                value_encoded_src = key_encoded_src.clone()

            pointer_attention_scores = attention_scores.clone().squeeze(-1)

            concat_out            = T.cat([h, context_vector], dim=-1)
            gen_dist_intermediate = F.dropout(self.out_linear1(concat_out), p=self.dropout, training=self.training)
            gen_dist              = F.softmax(self.out_linear2(gen_dist_intermediate), dim=-1)
            potential_extra_vocab = T.zeros(N, S1).float().to(src.device)
            gen_dist_extended     = T.cat([gen_dist, potential_extra_vocab], dim=-1)
            p_gen                 = T.sigmoid(self.pointer_linear(T.cat([input, context_vector, h], dim=-1)))

            assert gen_dist_extended.size()        == (N, self.vocab_len + S1)
            assert p_gen.size()                    == (N, 1)
            assert pointer_src_idx.size()          == (N, S1)
            assert pointer_attention_scores.size() == (N, S1)

            output_dist = (p_gen * gen_dist_extended).scatter_add(dim=-1, index=pointer_src_idx, src=((1.0 - p_gen) * pointer_attention_scores))
            output_dists.append(output_dist)


            prediction = T.argmax(output_dist, dim=-1, keepdim=False)
            input_idx  = T.where(prediction >= self.vocab_len, T.empty(N).fill_(self.UNK_id).long().to(src.device), prediction)
        
        logits     = T.stack(output_dists, dim=1)

        assert logits.size() == (N, S, self.vocab_len + S1)

        return {'logits': logits}