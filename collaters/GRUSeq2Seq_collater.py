import torch as T, numpy as np, random, copy, pdb

class GRUSeq2Seq_collater:
    def __init__(self, PAD, config, train):
        self.PAD       = PAD
        self.config    = config
        self.train     = train
        self.vocab_len = config['vocab_len']

    # pads <pad> token to all the items in the list, also calculates Mask as 
    # sequence of 1s where token was already present, 0s for new pads
    def pad(self, items, PAD):
        max_len = max([len(item) for item in items])

        padded_items, item_masks = [], []
        for item in items:
            mask = [1] * len(item)
            while len(item) < max_len:
                item.append(PAD)
                mask.append(0)
            padded_items.append(item)
            item_masks.append(mask)

        return padded_items, item_masks

    def sort_list(self, objs, idx):
        return [objs[i] for i in idx]

    def create_labels(self, trg_vec, src, trg):
        label = []
        while len(trg) < len(trg_vec):
            trg.append('<pad>')
        for token, id in zip(trg, trg_vec):
            if (id >= self.vocab_len) and (token in src):
                src_pos = 0
                for pos, token_ in enumerate(src):
                    if token_ == token:
                        src_pos = pos
                label.append(self.vocab_len + src_pos)
            else:
                label.append(id)

        assert len(label) == len(trg_vec)

        return label

    def collate_fn(self, batch):
        srcs     = [obj['src'] for obj in batch]
        trgs     = [obj['trg'] for obj in batch]
        srcs_vec = [obj['src_vec'] for obj in batch]
        trgs_vec = [obj['trg_vec'] for obj in batch]

        bucket_size = len(srcs)
        batch_size  = self.config['train_batch_size'] if self.train else self.config['dev_batch_size']
        
        lengths = [len(obj) for obj in srcs_vec]
        
        sorted_idx = np.argsort(lengths) # sorts in ascending order by default
        srcs       = self.sort_list(srcs, sorted_idx)
        trgs       = self.sort_list(trgs, sorted_idx)
        srcs_vec   = self.sort_list(srcs_vec, sorted_idx)
        trgs_vec   = self.sort_list(trgs_vec, sorted_idx)
        
        i = 0
        meta_batches = []
        while i < bucket_size:
            batches = []
            
            inr = batch_size
            if i + inr > bucket_size:
                inr = bucket_size - i

            inr_ = inr
            j = copy.deepcopy(i)
            while j < i + inr:
                srcs_vec_, src_masks = self.pad(srcs_vec[j:j+inr_], PAD=self.PAD)
                trgs_vec_, trg_masks = self.pad(trgs_vec[j:j+inr_], PAD=self.PAD)
                labels               = [self.create_labels(trg_vec=copy.deepcopy(trg_vec), src=copy.deepcopy(src), trg=copy.deepcopy(trg)) for trg_vec, src, trg in zip(trgs_vec_, srcs[j:j+inr_], trgs[j:j+inr_])]
                
                batch                      = {}
                batch['batch_size']        = inr_
                batch['src']               = srcs[j:j+inr_]
                batch['trg']               = trgs[j:j+inr_]
                batch['src_vec']           = T.tensor(srcs_vec_).long()
                batch['trg_vec']           = T.tensor(trgs_vec_).long()
                batch['src_mask']          = T.tensor(src_masks).float()
                batch['trg_mask']          = T.tensor(trg_masks).float()
                batch['labels']            = T.tensor(labels).long()

                batches.append(batch)
                j += inr_
            i += inr
            meta_batches.append(batches)

        random.shuffle(meta_batches)

        batches = []
        for batch_list in meta_batches:
            batches = batches + batch_list

        return batches
