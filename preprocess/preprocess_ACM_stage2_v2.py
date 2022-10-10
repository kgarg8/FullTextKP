# v2 - abstract <sep> title (without citations)
import json, shutil, pdb
from preprocess_tools.process_utils import jsonl_save
from pathlib import Path

def process_data(file, path):
    data = []
    with open('../processed_data/'+file) as f:
        for line in f:
            data.append(json.loads(line))

    SRC_MAX_LEN = 800
    srcs, src_vecs, trgs, trg_vecs = [], [], [], []

    for i in range(len(data)):
        # recreate src, src_vec
        indices = [i for i, x in enumerate(data[i]['src']) if x == '<sep>']
        src     = data[i]['src'][:indices[1]]
        src_vec = data[i]['src_vec'][:indices[1]]

        assert len(src) == len(src_vec)

        src     = src[:SRC_MAX_LEN]
        src_vec = src_vec[:SRC_MAX_LEN]

        # append to dictionary
        srcs.append(src)
        src_vecs.append(src_vec)
        trgs.append(data[i]['trg'])
        trg_vecs.append(data[i]['trg_vec'])
        
    data_dict = {'src': srcs, 'trg': trgs, 'src_vec': src_vecs, 'trg_vec': trg_vecs}
    jsonl_save(filepath=path+file, data_dict=data_dict)

dump_folder = '../stage2_processed_data/v2/'
Path(dump_folder).mkdir(parents=True, exist_ok=True)

process_data('ACM_train.jsonl'   , dump_folder)
process_data('ACM_dev_ACM.jsonl' , dump_folder)
process_data('ACM_test_ACM.jsonl', dump_folder)
shutil.copy('../processed_data/ACM_metadata.pkl', dump_folder)  # Copy metadata file as it is