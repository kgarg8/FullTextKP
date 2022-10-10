import json, jsonlines, numpy as np, time, torch, torch.nn as nn, sys, faiss, shutil, random, Levenshtein, pdb
from functools import reduce
from pathlib import Path
from tqdm import tqdm
sys.path.append('../preprocess/preprocess_tools')
from process_utils import jsonl_save

random.seed(137)

def loadTrainSentences():
    # Always load sentences & their embeddings from train set
    print('Sentence embeddings loading from train set...')

    embed_sent = []
    for line in open('embeddings/sentences_ACM_train.jsonl', 'r'):
        obj = json.loads(line)
        embed_sent.append(obj['embedding'])

    print('Sentences loading from train set...')
    
    sentences, sentences_vec = [], []
    with open('specter_format/sentences_ACM_train.jsonl') as fp:
        res = json.load(fp)
        for key in res.keys():
            sentences.append(res[key]['abstract'])    # sentence is stored in abstract field, title field is ""
            sentences_vec.append(res[key]['src_vec']) # sentence_vec is stored in src_vec

    assert len(sentences) == len(embed_sent)

    return embed_sent, sentences, sentences_vec

def process(path, file, ablation, embed_sent, sentences, sentences_vec):
    # Load article embeddings from embeddings folder and article src, src_vec from original processed folder
    # Compare the sentences against T+A of articles in train, test, dev set
    # Finally append similar sentences to T+A

    print('Article embeddings loading from {}...'.format(file))
    
    embed_TA = []
    for line in open('embeddings/articles_' + file, 'r'):
        obj = json.loads(line)
        embed_TA.append(obj['embedding'])

    # load articles, start populating T & A into src, keyphrases into trg
    print('Articles loading from {}...'.format(file))

    srcs, srcs_vec, trgs, trgs_vec = [], [], [], []
    for line in open('../processed_data/' + file, 'r'): # load previous processed_data
        data    = json.loads(line)
        src     = data['src']
        src_vec = data['src_vec']
        indices = [i for i, x in enumerate(src) if x == "<sep>"]
        
        try: # exception articles - just take the entire src as it is
            src     = src[:indices[1]]                  # just extract title and abstract
            src_vec = data['src_vec'][:indices[1]]
        except:     pass

        srcs.append(src)
        trgs.append(data['trg'])
        srcs_vec.append(src_vec)
        trgs_vec.append(data['trg_vec'])

    assert len(embed_TA) == len(srcs)

    sep_token_id = srcs_vec[0][srcs[0].index('<sep>')]
    tic          = time.perf_counter()

    print('Computing similarity using FAISS & preparing data...')

    dim   = len(embed_sent[0])
    index = faiss.IndexFlatL2(dim)

    if torch.cuda.is_available():
        res   = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(np.array(embed_sent, dtype =np.float32))  # index based on sentence embeddings
    
    src_deficiency = 0
    for i in tqdm(range(len(srcs))):
        
        num_queries = 100 # select only 60 sentences, 60*10 (avg length > 10) = at least 600 words (assuming T+A has around 200 words)

        if ablation:
            res_indices = random.sample(range(len(embed_sent)), num_queries)
        else:
            query          = np.array(embed_TA[i], dtype=np.float32)[None, :]    # query based on T+A embeddings
            _, res_indices = index.search(query, num_queries)
            res_indices    = res_indices.reshape(-1)

        dup_indices = []
        if file == 'ACM_train.jsonl':           # only train file will have duplicates
            # Remove sentences from the pool which are duplicates of those in own T+A
            # 1. create indices of '.' and <sep>
            src        = srcs[i]
            sep_index  = src.index('<sep>')
            indices    = [(i + sep_index) for i, x in enumerate(src[sep_index:]) if x == "."]
            indices    = [sep_index] + indices        # append first <sep> index, where abstract starts
            if src[-1] != '.':    indices.append(len(src)) # append end of abstract, either . or <sep>
            
            # 2. create self_pool of all sentences in T+A
            title       = ' '.join(src[:sep_index])
            self_pool   = [title] + [' '.join(src[indices[i]+1:indices[i+1]+1]) for i in range(0, len(indices)-1)]
            self_pool   = [item for item in self_pool if item != ''] # remove empty sentences

            # 3. collect sentences to be appended
            sa = [sentences[idx] for idx in res_indices]
            
            # 4. find indices of sentences that are also present in self_pool
            for j in range(len(self_pool)):
                l = [idx for idx, x in enumerate(sa) if self_pool[j] in x]
                dup_indices.extend(l)
            dup_indices = [res_indices[i] for i in dup_indices] # find corresponding index in res_indices

        # 5. finally, remove the duplicate sentences
        t1, t2 = [], []
        for idx in res_indices:
            if idx not in dup_indices:
                s1 = sentences[idx].split(' ')
                s1 = [item for item in s1 if item != '']
                s2 = sentences_vec[idx]
                t1.extend(['<sep>'] + s1)
                t2.extend([sep_token_id] + s2)

        assert len(t1) == len(t2)

        # append sentences to T+A
        srcs[i].extend(t1)
        srcs_vec[i].extend(t2)

        if len(srcs[i]) < 800:  src_deficiency += 1

        srcs[i]     = srcs[i][:800]
        srcs_vec[i] = srcs_vec[i][:800]

    toc = time.perf_counter()
    print(f'Similarity computed in {toc - tic:0.4f} seconds')
    print('Srcs deficiency count: {}'.format(src_deficiency))
    
    data_dict = {'src': srcs, 'trg': trgs, 'src_vec': srcs_vec, 'trg_vec': trgs_vec}
    jsonl_save(filepath=path+file, data_dict=data_dict)

ablation = False
if ablation:        path = 'processed_data/ablation/'
else:               path = 'processed_data/faiss/'

Path(path).mkdir(parents=True, exist_ok=True)

embed_sent, sentences, sentences_vec = loadTrainSentences()

process(path, 'ACM_train.jsonl'   , ablation, embed_sent, sentences, sentences_vec)
process(path, 'ACM_dev_ACM.jsonl' , ablation, embed_sent, sentences, sentences_vec)
process(path, 'ACM_test_ACM.jsonl', ablation, embed_sent, sentences, sentences_vec)
shutil.copy('../processed_data/ACM_metadata.pkl', path)  # Copy metadata file as it is