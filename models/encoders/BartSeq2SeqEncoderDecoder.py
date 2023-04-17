import torch as T
import torch.nn as nn
from transformers.models.bart.modeling_bart import BartForConditionalGeneration

class BartSeq2SeqEncoderDecoder(nn.Module):
    def __init__(self, config):

        super(BartSeq2SeqEncoderDecoder, self).__init__()

        self.embedding_path = config["embedding_path"]
        self.model = BartForConditionalGeneration.from_pretrained(self.embedding_path, gradient_checkpointing=True, return_dict=True)
        self.config = config

    # %%

    def forward(self, src_idx, input_mask, trg_idx=None, output_mask=None):

        N, S = src_idx.size()

        if not self.config['generate']:
            outputs = self.model(input_ids=src_idx,
                                 use_cache=False,
                                 labels=trg_idx,
                                 attention_mask=input_mask,
                                 decoder_attention_mask=output_mask)

            logits = outputs.logits

            prediction = T.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
        else:
            outputs = self.model.generate(input_ids=src_idx,
                                          use_cache=True,
                                          num_beams=self.config["num_beams"],
                                          max_length=50,
                                          attention_mask=input_mask,
                                          num_return_sequences=self.config["num_beams"])

            prediction = outputs
            prediction = prediction.view(N, self.config["num_beams"], -1)
            prediction = prediction.detach().cpu().numpy().tolist()

            logits = None

        return {"logits": logits, "prediction": prediction}
