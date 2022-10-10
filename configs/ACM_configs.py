from os import fspath
from pathlib import Path

class optimizer_config:
    def __init__(self):
        # optimizer config
        self.max_grad_norm       = 1.0
        self.bucket_size_factor  = 1
        self.DataParallel        = True
        self.weight_decay        = 0.0
        self.epochs              = 1
        self.early_stop_patience = 2
        self.scheduler_patience  = 0
        self.warm_up_steps       = 2000
        self.save_by             = "loss"
        self.metric_direction    = -1
        self.validation_interval = 1
        self.num_workers         = 6


class base_config(optimizer_config):
    def __init__(self):
        super().__init__()
        self.word_embd_freeze     = False # word embedding
        self.embd_dim             = 100 # hidden size
        self.teacher_force_ratio  = 1.0
        self.encoder_hidden_size  = 150
        self.encoder_layers       = 1
        self.decoder_hidden_size  = 300
        self.coverage_mechanism   = False
        self.max_decoder_len      = 60
        self.disentangled_pointer = False
        self.key_value_attention  = False


class GRUSeq2Seq_config(base_config):
    def __init__(self):
        super().__init__()
        self.model_name       = "(GRU Seq2Seq)"
        self.encoder          = "GRUEncoderDecoder"
        self.optimizer        = "Adam"
        self.lr               = 1e-3
        self.epochs           = 20
        self.batch_size       = 12
        self.train_batch_size = 12
        self.dev_batch_size   = 32
        self.chunk_size       = self.batch_size * 4000
        self.dropout          = 0.1

class LEDSeq2Seq_config(base_config):
    def __init__(self):
        super().__init__()
        self.model_name         = "(LED Seq2Seq)"
        self.embedding_path     = fspath(Path("embeddings/led_base_16384"))
        self.optimizer          = "Adam"
        self.lr                 = 5*1e-5
        self.encoder            = "LEDSeq2SeqEncoderDecoder"
        self.epochs             = 10
        self.batch_size         = 12
        self.train_batch_size   = 12
        self.dev_batch_size     = 16
        self.chunk_size         = self.batch_size * 4000
        self.num_beams          = 1
        self.bucket_size_factor = 1