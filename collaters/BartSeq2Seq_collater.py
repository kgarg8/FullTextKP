import torch as T, copy, numpy as np, random
from transformers.models.bart.tokenization_bart import BartTokenizer

class BartSeq2Seq_collater:
    def __init__(self, PAD, config, train):
        self.PAD = PAD
        self.config = config
        self.tokenizer = BartTokenizer.from_pretrained(config["embedding_path"])
        self.sep_token_id = self.tokenizer.encode("<sep>", add_special_tokens=False)[0]
        self.train = train

    def pad(self, items, PAD):
        max_len = max([len(item) for item in items])

        padded_items = []
        item_masks = []
        for item in items:
            mask = [1]*len(item)
            while len(item) < max_len:
                item.append(PAD)
                mask.append(0)
            padded_items.append(item)
            item_masks.append(mask)

        return padded_items, item_masks

    def sort_list(self, objs, idx):
        return [objs[i] for i in idx]

    def collate_fn(self, batch):
        
        # if self.config['dataset'] == 'ACM' or self.config['dataset'] == 'KPTimes' or self.config['dataset'] == 'KPTimes800' or self.config['dataset'] == 'OpenKP800': # ACM not tokenized in preprocessing, tokenize here
        SRC_MAX_LEN = 800
        for obj in batch:
            obj['src'] = obj['src'][:SRC_MAX_LEN] # truncate src at SRC_MAX_LEN boundary
            tokenized_src = self.tokenizer.encode(' '.join(obj['src']), truncation=True, max_length=1024)
            trg_split = ' '.join(obj['trg']).split(' <eos>')[0].split(' ; ') # extra spaces given for ';', '<eos>' remove whitespaces
            
            tokenized_trg_split = [self.tokenizer.encode(kp, add_special_tokens=False) for kp in trg_split]
            tokenized_trg = []

            for i, tokenized_kp in enumerate(tokenized_trg_split):
                if i != 0:
                    tokenized_trg += [self.sep_token_id] + tokenized_kp
                else:
                    tokenized_trg = tokenized_kp

            tokenized_trg = tokenized_trg + [self.tokenizer.eos_token_id]
            
            obj['tokenized_src'] = tokenized_src
            obj['tokenized_trg'] = tokenized_trg

        tokenized_srcs = [obj['tokenized_src'] for obj in batch] # don't have to truncate in case of Bart
        tokenized_trgs = [obj['tokenized_trg'] for obj in batch]
        srcs = [obj['src'] for obj in batch]
        trgs = [obj['trg'] for obj in batch]

        bucket_size = len(srcs)
        batch_size  = self.config['train_batch_size'] if self.train else self.config['dev_batch_size']
        
        lengths    = [len(obj) for obj in tokenized_srcs]
        sorted_idx = np.argsort(lengths) # sorts in ascending order by default
        
        srcs           = self.sort_list(srcs, sorted_idx)
        trgs           = self.sort_list(trgs, sorted_idx)
        tokenized_srcs = self.sort_list(tokenized_srcs, sorted_idx)
        tokenized_trgs = self.sort_list(tokenized_trgs, sorted_idx)

        i = 0
        meta_batches = []
        while i < bucket_size:
            batches = []
            
            inr = batch_size
            if i + inr > bucket_size:
                inr = bucket_size - i

            max_len1 = max([len(obj) for obj in tokenized_srcs[i:i + inr]])
            max_len2 = max([len(obj) for obj in tokenized_trgs[i:i + inr]])

            # Ensure batch_size of minimum 8 for slicing to work properly
            assert (batch_size >= 8)
            if max_len1 >= 2500 or max_len2 >= 100:
                inr_ = min(batch_size // 8, inr)
            elif max_len1 >= 1000:
                inr_ = min(batch_size // 4, inr)
            elif max_len1 >= 500:
                inr_ = min(batch_size // 2, inr)
            else:
                inr_ = inr
            
            j = copy.deepcopy(i)
            while j < i + inr:
                srcs_vec, src_masks = self.pad(tokenized_srcs[j:j+inr_], PAD=self.PAD)
                trgs_vec, trg_masks = self.pad(copy.deepcopy(tokenized_trgs[j:j+inr_]), PAD=self.PAD)
                labels, _           = self.pad(tokenized_trgs[j:j+inr_], PAD=-100)

                batch               = {}
                batch['batch_size'] = inr_
                batch['src']        = srcs[j:j+inr_]
                batch['trg']        = trgs[j:j+inr_]
                batch['src_vec']    = T.tensor(srcs_vec).long()
                batch['trg_vec']    = T.tensor(trgs_vec).long()
                batch['src_mask']   = T.tensor(src_masks).float()
                batch['trg_mask']   = T.tensor(trg_masks).float()
                batch['labels']     = T.tensor(labels).long()
                batches.append(batch)
                j += inr_
            i += inr
            meta_batches.append(batches)

        random.shuffle(meta_batches)

        batches = []
        for batch_list in meta_batches:
            batches = batches + batch_list

        return batches
