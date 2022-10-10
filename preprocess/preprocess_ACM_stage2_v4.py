# v4 - abstract <sep> title <sep> citation_context [order preservation random selection]
import json, shutil, random, pdb
from preprocess_tools.process_utils import jsonl_save
from pathlib import Path

random.seed(137)

def process_data(file, path):
    def word_count(text):
        return sum([len(element) for element in text])

    data = []
    with open('../processed_data/'+file) as f:
        for line in f:
            data.append(json.loads(line))

    SRC_MAX_LEN = 800
    citation_idxs_list, srcs, src_vecs, trgs, trg_vecs = [], [], [], [], []

    for i in range(len(data)):
        citation_idxs     = [idx for idx, item in enumerate(data[i]['tag']) if item == 'C']
        citation_idxs_list.append(citation_idxs)
        
        # recreate src, src_vec
        src     = data[i]['src']
        src_vec = data[i]['src_vec']

        indices = [i for i, x in enumerate(src) if x == '<sep>']
        indices.append(len(src))

        sentences, sentences_vec      = [], []
        for idx in range(1, len(indices) - 1):                        # start from second <sep> to leave out title & abstract
            sentences.append(src[indices[idx]+1:indices[idx+1]])      # extract sentences leaving <out> sep tokens
            sentences_vec.append(src_vec[indices[idx]+1:indices[idx+1]])

        cite_sentences     = [sentences[idx] for idx in citation_idxs]
        cite_sentences_vec = [sentences_vec[idx] for idx in citation_idxs]

        title_abstract     = src[:indices[1]]
        title_abstract_vec = src_vec[:indices[1]]
        src                = title_abstract
        src_vec            = title_abstract_vec
        sep_token_id       = src_vec[indices[0]]
        src_len            = sum([len(element) + 1 for element in cite_sentences]) + len(src[:indices[1]])
        
        if src_len > SRC_MAX_LEN:
            
            if cite_sentences != []:
                # recreate src (list of words) of max length SRC_MAX_LEN
                # 1) keep title & abstract as it is
                # 2) order preserved random selection for citation sentences

                scarce_factor = SRC_MAX_LEN - len(src)
                sublist_idxs  = []
                idxs          = [*range(len(cite_sentences))]
                counter       = 0
                while(1):
                    counter += 1
                    if counter >= 8:    break
                    rand_idxs    = random.sample(idxs, min(len(idxs), 15))
                    sublist_idxs += rand_idxs       # simply append, sort later
                    idxs         = list(set(idxs) - set(rand_idxs))
                    text         = [cite_sentences[i] for i in sublist_idxs]
                    if word_count(text) + len(src) > scarce_factor:
                        rand_idxs    = random.sample(idxs, min(len(idxs), 15))
                        sublist_idxs += rand_idxs       # simply append, sort later
                        break
                
                sublist_idxs = sorted(sublist_idxs)
                for idx in sublist_idxs:
                    src.append('<sep>')
                    src_vec.append(sep_token_id)
                    src.extend(cite_sentences[idx])
                    src_vec.extend(cite_sentences_vec[idx])
                
        else:   # simply append citations to title and abstract
            src          = src[:indices[1]]
            src_vec      = src_vec[:indices[1]]
            sep_token_id = src_vec[indices[0]]
            for idx in range(len(cite_sentences)):
                src.append('<sep>')
                src_vec.append(sep_token_id)
                src.extend(cite_sentences[idx])
                src_vec.extend(cite_sentences_vec[idx])

        assert len(src) == len(src_vec)
        src     = src[0:SRC_MAX_LEN]
        src_vec = src_vec[0:SRC_MAX_LEN]
        
        # append to dictionary
        srcs.append(src)
        src_vecs.append(src_vec)
        trgs.append(data[i]['trg'])
        trg_vecs.append(data[i]['trg_vec'])
        
    data_dict = {'src': srcs, 'trg': trgs, 'src_vec': src_vecs, 'trg_vec': trg_vecs}
    jsonl_save(filepath=path+file, data_dict=data_dict)

dump_folder = '../stage2_processed_data/v4/'
Path(dump_folder).mkdir(parents=True, exist_ok=True)

process_data('ACM_train.jsonl'   , dump_folder)
process_data('ACM_dev_ACM.jsonl' , dump_folder)
process_data('ACM_test_ACM.jsonl', dump_folder)
shutil.copy('../processed_data/ACM_metadata.pkl', dump_folder)  # Copy metadata file as it is
