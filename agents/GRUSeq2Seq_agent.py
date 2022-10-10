import copy, math, nltk, numpy as np, torch as T, torch.nn as nn, torch.nn.functional as F, pdb
from controllers.optimizer_controller import get_optimizer
from utils.evaluation_utils import evaluate
from nltk.stem import PorterStemmer

class GRUSeq2Seq_agent:
    def __init__(self, model, vocab2idx, config, device):
        self.model                 = model
        self.parameters            = [p for p in model.parameters() if p.requires_grad]
        optimizer                  = get_optimizer(config)
        self.optimizer             = optimizer(self.parameters, lr=config['lr'], weight_decay=config['weight_decay'])
        self.config                = config
        self.key                   = 'none'
        self.device                = device
        self.DataParallel          = config['DataParallel']
        self.idx2vocab             = {id: token for token, id in vocab2idx.items()}
        self.vocab2idx             = vocab2idx
        self.vocab_len             = len(vocab2idx)
        self.eps                   = 1e-9
        self.epoch_level_scheduler = True
        self.scheduler             = T.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=config['scheduler_patience'])
        self.optimizer.zero_grad()

    def loss_fn(self, logits, labels, output_mask):
        
        vocab_len                 = logits.size(-1)
        N, S                      = labels.size()
        assert logits.size()      == (N, S, vocab_len)
        assert output_mask.size() == (N, S)
        assert (logits >= 0.0).all()
        assert (logits <= 1.0).all()

        true_dist               = F.one_hot(labels, num_classes=vocab_len)
        assert true_dist.size() == (N, S, vocab_len)
        assert (true_dist >= 0).all()

        logits = T.where(logits==0.0, T.empty_like(logits).fill_(self.eps).float().to(logits.device), logits)
        
        neg_log_logits         = -T.log(logits)
        assert (neg_log_logits >= 0).all()
        assert true_dist.size() == neg_log_logits.size()

        cross_entropy               = T.sum(neg_log_logits * true_dist, dim=-1)
        assert cross_entropy.size() == (N, S)
        
        masked_cross_entropy = cross_entropy * output_mask
        mean_ce              = T.sum(masked_cross_entropy, dim=1) / (T.sum(output_mask, dim=1) + self.eps)
        loss                 = T.mean(mean_ce)
        assert loss >= 0.0

        return loss

    def decode(self, prediction_idx, src):
        decoded_prediction = []
        for id in prediction_idx:
            if id >= self.vocab_len:
                decoded_prediction.append(src[id - self.vocab_len])
            else:
                decoded_prediction.append(self.idx2vocab[id])
        return ' '.join(decoded_prediction)

    def run(self, batch, train=True):
        self.model   = self.model.train() if train else self.model.eval()
        output_dict  = self.model(batch)
        logits       = output_dict['logits']

        if logits is not None and not self.config['generate']:
            labels = batch['labels'].to(logits.device)
            loss   = self.loss_fn(logits=logits, labels=labels.to(logits.device), output_mask=batch['trg_mask'].to(logits.device))
        else:
            loss = None

        predictions = T.argmax(logits, dim=-1)
        predictions = predictions.cpu().detach().numpy().tolist()
        
        predictions = [self.decode(prediction, src) for prediction, src in zip(predictions, batch['src'])]

        metrics = evaluate(copy.deepcopy(batch['src']), copy.deepcopy(batch['trg']), copy.deepcopy(predictions), beam=False, LED_model=False, key=self.key)

        metrics['loss'] = 0.0
        if loss is not None:
            metrics['loss'] = loss.item()

        item = {'display_items': {'source': batch['src'], 'target': batch['trg'], 'predictions': predictions}, 'loss': loss, 'metrics': metrics, 'stats_metrics': metrics}
        return item

    def backward(self, loss):
        loss.backward()

    def step(self):
        if self.config['max_grad_norm'] is not None:
            T.nn.utils.clip_grad_norm_(self.parameters, self.config['max_grad_norm'])
        self.optimizer.step()
        self.optimizer.zero_grad()
