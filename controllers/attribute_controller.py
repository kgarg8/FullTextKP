def prepare_attributes(data, args):
    attributes = {"vocab_len": data["vocab_len"],
                  "pad_id": data["PAD_id"],
                  "unk_id": data["UNK_id"],
                  "sep_id": data["SEP_id"]}

    return attributes