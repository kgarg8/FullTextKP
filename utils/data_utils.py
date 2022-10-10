import json, pdb
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def load_dataset(path, limit=-1, keys=None):
    if limit == 0 or limit < -1:
        raise ValueError("limit must be either -1 or a positive integer")

    samples = {}
    idx = 0
    with open(path, 'r') as f:
        for line in f:
            if limit != -1 and idx + 1 == limit:
                    break
            samples[idx] = json.loads(line)
            idx += 1

    return Dataset(samples)

def load_data(paths, metadata, args):
    
    data = {}
    data['vocab2idx'] = metadata['vocab2idx']

    if data['vocab2idx'] is not None:
        vocab2idx = data['vocab2idx']
        data['PAD_id']    = vocab2idx['<pad>']
        data['UNK_id']    = vocab2idx['<unk>'] if '<unk>' in vocab2idx else None
        data['SEP_id']    = vocab2idx['<sep>'] if '<sep>' in vocab2idx else None
        data['vocab_len'] = len(vocab2idx)
        data['idx2vocab'] = {token: id for token, id in vocab2idx.items()}
    else:
        data["PAD_id"]    = None
        data["SEP_id"]    = None
        data["UNK_id"]    = None
        data["vocab_len"] = None
        data["idx2vocab"] = None
    
    data['train'] = load_dataset(paths['train'], limit=args.limit)
    data['dev']   = {key: load_dataset(paths['dev'][key], limit=args.limit) for key in paths['dev']}
    data['test']  = {key: load_dataset(paths['test'][key], limit=args.limit) for key in paths['test']}
    return data

def load_dataloaders(train_batch_size, dev_batch_size, partitions, train_collater_fn, dev_collater_fn, num_workers=6):

    dataloaders = {}
    dataloaders['train'] = DataLoader(Dataset(partitions['train']), batch_size=train_batch_size, num_workers=num_workers, shuffle=True, collate_fn=train_collater_fn)

    for split_key in ['dev', 'test']:
        dataloaders[split_key] = {key: DataLoader(Dataset(partitions[split_key][key]), batch_size=dev_batch_size, num_workers=num_workers, shuffle=False, collate_fn=dev_collater_fn) for key in partitions[split_key]}

    return dataloaders

def count_iterations(data_len, batch_size):
    iters = data_len // batch_size
    if data_len % batch_size > 0:
        iters += 1
    return iters
