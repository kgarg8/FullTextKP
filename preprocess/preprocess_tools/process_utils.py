import jsonlines

def jsonl_save(filepath, data_dict):
    for key in data_dict:
        data_len = len(data_dict[key])
        break
    objs = []
    for i in range(data_len):
        obj = {key: data_dict[key][i] for key in data_dict}
        objs.append(obj)
    with jsonlines.open(filepath, mode='w') as writer:
        writer.write_all(objs)
