import torch as T, torch.nn as nn, torch.nn.functional as F
from models.layers import Linear

class attention(nn.Module):
    def __init__(self, config):
        super(attention, self).__init__()
        self.config = config
        self.encoder_hidden_size = config['encoder_hidden_size']
        self.decoder_hidden_size = config['decoder_hidden_size']
        self.attn_linear1 = Linear(2 * self.encoder_hidden_size + self.decoder_hidden_size, self.decoder_hidden_size)
        self.attn_linear2 = Linear(self.decoder_hidden_size, 1)
        self.eps = 1e-9

    def forward(self, key_encoder_states, value_encoder_states, decoder_state, attention_mask, input_mask):
        N, S, _          = key_encoder_states.size()
        decoder_state    = decoder_state.unsqueeze(1).repeat(1, S, 1)
        energy           = self.attn_linear2(T.tanh(self.attn_linear1(T.cat([key_encoder_states, decoder_state], dim=-1))))
        attention_scores = F.softmax(energy + attention_mask, dim=1)*input_mask.unsqueeze(-1)

        assert (attention_scores <= 1.0).all()
        context_vector = T.sum(attention_scores * value_encoder_states, dim=1)

        return attention_scores, context_vector, key_encoder_states
