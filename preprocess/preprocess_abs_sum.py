import nltk, re, json, numpy as np, pickle, pdb
from tqdm import tqdm
from nltk.stem import PorterStemmer
from pathlib import Path
from preprocess_tools.process_utils import jsonl_save
from os import fspath


dataset     = 'ACM' # doesn't contain any citation information but only abstract & title and keywords
vocab2count = {}
tokenizer   = nltk.tokenize.MWETokenizer(separator='') # separator is inserted between multiple words when combining
tokenizer.add_mwe(('<','digit','>')) # strings to combine - <, digit, >
tokenizer.add_mwe(('<', 'sep', '>'))
tokenizer.add_mwe(('<', 'eos', '>'))
tokenizer.add_mwe(('<', 'pad', '>'))
tokenizer.add_mwe(('<', 'unk', '>'))


# order the keyphrases - present KP followed by absent KP, present KP aligned acc to their occurence in src
def order_keyphrases(src, keyphrases):
    ps = PorterStemmer()
    src = [ps.stem(word) for word in src] # stem src
    src = ' '.join(src)

    # stem keyphrases & compare to stemmed src
    stemmed_keyphrases = [[ps.stem(word) for word in nltk.word_tokenize(phrase)] for phrase in keyphrases]
    match_indices = [src.find(' '.join(kp)) for kp in stemmed_keyphrases]
    sorted_indices = np.argsort(match_indices)
    
    present_keyphrases, absent_keyphrases = [], []
    for i in range(len(match_indices)):
        if match_indices[sorted_indices[i]] != -1:
            present_keyphrases.append(keyphrases[sorted_indices[i]])
        else:
            absent_keyphrases.append(keyphrases[sorted_indices[i]])
    
    present_keyphrases.extend(absent_keyphrases)
    return present_keyphrases


def process_data(processed_articles, update_vocab=False):
    print('Processing articles...')

    srcs, trgs = [], []
    for _, article in tqdm(enumerate(processed_articles)):
        src = article['src']
        src = nltk.word_tokenize(src) # word_tokenize is better than simple split, splits at punctuation also
        src = tokenizer.tokenize(src) # merge tokens with multi-words
        srcs.append(src)
        ordered_keyphrases = order_keyphrases(src, article['keywords'])
        
        trg = []
        for keyword in ordered_keyphrases:
            trg.append(keyword)
            trg.append(';')
        trg.pop()
        trg.append('<eos>')
        trg = nltk.word_tokenize(' '.join(trg))
        trg = tokenizer.tokenize(trg)
        trgs.append(trg)

        if update_vocab:
            for word in src:
                vocab2count[word] = vocab2count.get(word, 0) + 1
            for word in trg:
                vocab2count[word] = vocab2count.get(word, 0) + 1

    srcs_dict    = {}
    srcs_, trgs_, tags_ = [], [], []
    duplicate_count = full_duplicate_count = zero_trg_count = 0

    for src, trg in zip(srcs, trgs):
        flag = 0
        src_string = ' '.join(src)
        trg_string = ' '.join(trg)

        if src_string not in srcs_dict:
            srcs_dict[src_string] = [trg_string]
            flag = 0
        else:
            flag = 0
            duplicate_count += 1
            if trg_string in srcs_dict[src_string]:
                full_duplicate_count += 1
                flag = 1
            else:
                srcs_dict[src_string].append(trg_string)

        if len(trg) == 0:
            zero_trg_count += 1
            if train:
                flag = 1 # dont append if no keyphrases OR if duplicate article

        if flag == 0:
            srcs_.append(src)
            trgs_.append(trg)

    srcs, trgs = srcs_, trgs_

    print("Duplicate Count: {}, Full Duplicate Count: {}, Zero Trg Count: {}".format(duplicate_count, full_duplicate_count, zero_trg_count))
    assert len(srcs) == len(trgs)
    return srcs, trgs


def text_vectorize(text):
    return [vocab2idx.get(word, vocab2idx['<unk>']) for word in text]


def vectorize_data(srcs, trgs):
    data_dict            = {}
    srcs_vec             = [text_vectorize(src) for src in srcs]
    trgs_vec             = [text_vectorize(trg) for trg in trgs]
    data_dict['src']     = srcs
    data_dict['trg']     = trgs
    data_dict['src_vec'] = srcs_vec
    data_dict['trg_vec'] = trgs_vec
    return data_dict


def get_data(mode):

    with open('../abstractive_summarization/abs_summary_stage1_{}.json'.format(mode)) as f:
        data = json.load(f)

    articles = []
    for i in tqdm(range(len(data['src']))):
        article = {}
        article['src'] = data['src'][i]                                         # text
        article['keywords'] = data['trg'][i].split(' <eos>')[0].split(' ; ')    # keyphrases
        articles.append(article)

    print(len(articles))
    return articles


train_articles         = get_data('train')
train_srcs, train_trgs = process_data(train_articles, update_vocab=True)
dev_articles           = get_data('dev')
dev_srcs, dev_trgs     = process_data(dev_articles, update_vocab=True)
test_articles          = get_data('test')
test_srcs, test_trgs   = process_data(test_articles, update_vocab=False)

counts, vocab = [], []
for word, count in vocab2count.items():
    vocab.append(word)
    counts.append(count)

print('Total vocab size: ', len(vocab))
count = len([i for i in counts if i < 5])
print('Words with count less than 5: ', count)

MAX_VOCAB  = 100000
sorted_idx = np.flip(np.argsort(counts), axis=0) # indices of more frequent to less frequent words
vocab = [vocab[id] for id in sorted_idx]
if len(vocab) > MAX_VOCAB:
    vocab = vocab[:MAX_VOCAB]

vocab2idx = {}
for i, token in enumerate(vocab):
    vocab2idx[token] = i

special_tags = [";", "<sep>", "<unk>", "<eos>", "<peos>", "<pad>", "<digit>"]
for token in special_tags:
    if token not in vocab2idx:
        vocab2idx[token] = len(vocab2idx)

train_data = vectorize_data(train_srcs, train_trgs)
dev_data   = vectorize_data(dev_srcs, dev_trgs)
test_data  = vectorize_data(test_srcs, test_trgs)

Path('../processed_data/abs_sum_stage2').mkdir(parents=True, exist_ok=True)
jsonl_save(filepath='../processed_data/abs_sum_stage2/{}_train_{}.jsonl'.format(dataset, dataset), data_dict=train_data)
jsonl_save(filepath='../processed_data/abs_sum_stage2/{}_dev_{}.jsonl'.format(dataset, dataset), data_dict=dev_data)
jsonl_save(filepath='../processed_data/abs_sum_stage2/{}_test_{}.jsonl'.format(dataset, dataset), data_dict=test_data)

metadata = {'dev_keys': [dataset], 'test_keys': [dataset], 'vocab2idx': vocab2idx}
metadata_save_path = fspath(Path('../processed_data/abs_sum_stage2/{}_metadata.pkl'.format(dataset)))

with open(metadata_save_path, 'wb') as outfile:
    pickle.dump(metadata, outfile)