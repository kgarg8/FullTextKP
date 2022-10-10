import torch as T
import torch.nn as nn
from transformers.models.led.tokenization_led import LEDTokenizer

from controllers.optimizer_controller import get_optimizer
from utils.evaluation_utils import evaluate
import copy
import math
import nltk
from nltk.stem import PorterStemmer
import numpy as np


class LEDSeq2Seq_agent:
    def __init__(self, model, vocab2idx, config, device):
        self.model = model
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = get_optimizer(config)
        self.optimizer = optimizer(self.parameters,
                                   lr=config["lr"],
                                   weight_decay=config["weight_decay"])
        self.scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=config['scheduler_patience'])
        self.config = config
        self.device = device
        self.stemmer = PorterStemmer()
        self.DataParallel = config["DataParallel"]
        self.optimizer.zero_grad()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        self.tokenizer = LEDTokenizer.from_pretrained(config["embedding_path"])
        self.epoch_level_scheduler = True

    def pad(self, items, PAD):
        max_len = max([len(item) for item in items])

        padded_items = []
        item_masks = []
        for item in items:
            mask = [1] * len(item)
            while len(item) < max_len:
                item.append(PAD)
                mask.append(0)
            padded_items.append(item)
            item_masks.append(mask)

        return padded_items, item_masks

    def run(self, batch, train=True):

        if train:
            self.model = self.model.train()
        else:
            self.model = self.model.eval()

        output_dict = self.model(batch)
        logits = output_dict["logits"]
        predictions = output_dict["prediction"]

        if not self.config["generate"]:
            predictions = [self.tokenizer.decode(prediction[1:]) for prediction in predictions]
        else:
            predictions_ = []
            for beam_prediction in predictions:
                beam_prediction = [self.tokenizer.decode(prediction[1:]) for prediction in beam_prediction]
                predictions_.append(beam_prediction)
            predictions = predictions_

        if logits is not None:
            labels = batch["labels"].to(logits.device)

            N = logits.size(0)
            S = logits.size(1)

            logits = logits.view(N * S, -1)
            labels = labels.view(-1)

            loss = self.criterion(logits, labels)

        else:
            loss = None

        if not self.config["generate"]:
            metrics = evaluate(batch["src"], batch["trg"], predictions, beam=False, LED_model=True)
        else:
            predictions_ = [beam_prediction[0] for beam_prediction in predictions]
            metrics = evaluate(copy.deepcopy(batch["src"]), copy.deepcopy(batch["trg"]), copy.deepcopy(predictions_),
                               beam=False, LED_model=True)
            metrics_beam = evaluate(batch["src"], batch["trg"], predictions, beam=True, LED_model=True)
            for key in metrics_beam:
                metrics[key + "_beam"] = metrics_beam[key]

        if loss is not None:
            metrics["loss"] = loss.item()
        else:
            metrics["loss"] = 0.0

        item = {"display_items": {"source": batch["src"],
                                  "target": batch["trg"],
                                  "predictions": predictions},
                "loss": loss,
                "metrics": metrics,
                "stats_metrics": metrics}

        return item

    def backward(self, loss):
        loss.backward()

    def step(self):
        if self.config["max_grad_norm"] is not None:
            T.nn.utils.clip_grad_norm_(self.parameters, self.config["max_grad_norm"])
        self.optimizer.step()
        self.optimizer.zero_grad()
