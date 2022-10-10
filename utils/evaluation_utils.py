import copy, random, nltk, pdb
from nltk.stem import PorterStemmer

def process_predictions(predictions, LED_model, beam=False):
    stemmer = PorterStemmer()
    processed_predictions = []

    for beam_prediction in predictions:
        
        eos_token = "<eos>"
        if LED_model:
            eos_token = "</s>"

        if beam:
            prediction_ = ""
            for prediction in beam_prediction:
                prediction = prediction.split(eos_token)[0]
                if not prediction_:
                    prediction_ += prediction
                else:
                    prediction_ += ' <sep> ' + prediction
            prediction = prediction_
        else:
            prediction = beam_prediction.split(eos_token)[0]
        # broadly - splitting the prediction into keyphrases to form a list
        # split() converts a string into list
        prediction = prediction.replace(';', '<sep>').replace('<unk>', '').split("<sep>")

        # get the list of stemmed keyphrases
        stemed_prediction = []
        for kp in prediction:
            kp = kp.strip()
            if kp != "":  # and "." not in kp and "," not in kp
                tokenized_kp        = kp.split(" ")  # nltk.word_tokenize(kp)
                tokenized_stemed_kp = [stemmer.stem(kw).strip() for kw in tokenized_kp]
                stemed_kp           = " ".join(tokenized_stemed_kp).replace("< digit >", "<digit>")
                if stemed_kp.strip() != "":
                    stemed_prediction.append(stemed_kp.strip())

        # make prediction duplicates free but preserve order for @topk
        prediction_dict = {}
        stemed_prediction_ = []
        for kp in stemed_prediction:
            if kp not in prediction_dict:
                prediction_dict[kp] = 1
                stemed_prediction_.append(kp)
        stemed_prediction = stemed_prediction_

        processed_predictions.append(stemed_prediction)

    return processed_predictions


def process_ground_truths(trgs, key="none"):
    stemmer = PorterStemmer()
    processed_trgs = []
    for trg in trgs:
        # trg is tokenized list; join basically merges those tokens to form a string
        # finally get a list of keyphrases
        trg_split  = " ".join(trg).replace(";", " <sep> ").split("<eos>")[0].split("<sep>")
        stemed_trg = []
        
        for kp in trg_split:
            tokenized_kp = kp.split(" ")
            
            if key != "semeval": # semeval already has stemmed words
                tokenized_stemed_kp = [stemmer.stem(kw).strip() for kw in tokenized_kp]
            else:
                tokenized_stemed_kp = [kw.strip() for kw in tokenized_kp]
            stemed_kp = " ".join(tokenized_stemed_kp).replace("< digit >", "<digit>")
        
            if stemed_kp.strip() != "":
                stemed_trg.append(stemed_kp.strip())
        
        # make duplicates free w/o preserving the order, set() doesn't preserve order
        stemed_trg = list(set(stemed_trg))
        processed_trgs.append(stemed_trg)

    return processed_trgs

def process_srcs(srcs):
    stemmer = PorterStemmer()
    processed_srcs = []
    for src in srcs:
        tokenized_src        = src
        tokenized_stemed_src = [stemmer.stem(token).strip() for token in tokenized_src]
        stemed_src           = " ".join(tokenized_stemed_src).strip().replace("< digit >", "<digit>")
        processed_srcs.append(stemed_src)
    return processed_srcs

def evaluate(srcs, trgs, predictions, beam=False, LED_model=False, key="none"):
    assert len(srcs) == len(trgs) == len(predictions)
    
    # 1 prediction -> stemmed keyphrases list, we have multiple such predictions
    predictions = process_predictions(predictions, LED_model, beam)
    trgs        = process_ground_truths(trgs, key)
    srcs        = process_srcs(srcs)

    total_present_precision, total_present_recall, total_absent_precision, total_absent_recall = {}, {}, {}, {}

    i = 0
    total_data = 0
    for src, trg, prediction in zip(srcs, trgs, predictions):
        
        src = " <sep> ".join(src.split(" <sep> ", 2)[:2]) # hack for removing citations before calculating present & absent KP
        present_trg, absent_trg = [], []
        for kp in trg:
            present_trg.append(kp) if kp in src else absent_trg.append(kp)
            
        present_prediction, absent_prediction = [], []
        for kp in prediction:
            present_prediction.append(kp) if kp in src else absent_prediction.append(kp)

        present_precision, present_recall, absent_precision, absent_recall = {}, {}, {}, {}
        original_present_prediction = copy.deepcopy(present_prediction)
        original_absent_prediction  = copy.deepcopy(absent_prediction)

        for topk in ["5R", "M"]:
            present_prediction = copy.deepcopy(original_present_prediction)
            absent_prediction  = copy.deepcopy(original_absent_prediction)
            if topk == "M":
                pass
            elif topk == "O":
                if not present_trg:
                    present_prediction = []
                elif len(present_prediction) > len(present_trg):
                    present_prediction = present_prediction[0:len(present_trg)]
                if not absent_trg:
                    absent_prediction = []
                elif len(absent_prediction) > len(absent_trg):
                    absent_prediction = absent_prediction[0:len(absent_trg)]
            else:
                if "R" in topk:
                    R    = True
                    topk = int(topk[0:-1])
                else:
                    R    = False
                    topk = int(topk)
                if len(present_prediction) > topk:
                    present_prediction = present_prediction[0:topk]
                elif R:
                    while len(present_prediction) < topk:
                        present_prediction.append("<fake keyphrase>")
                if len(absent_prediction) > topk:
                    absent_prediction = absent_prediction[0:topk]
                elif R:
                    while len(absent_prediction) < topk:
                        absent_prediction.append("<fake keyphrase>")
                topk = str(topk)
                if R:
                    topk = topk + "R"

            present_tp = 0
            for kp in present_prediction:
                if kp in present_trg:
                    present_tp += 1

            present_precision[topk] = present_tp / len(present_prediction) if len(present_prediction) > 0 else 0
            present_recall[topk]    = present_tp / len(present_trg) if len(present_trg) > 0 else 0

            absent_tp = 0
            for kp in absent_prediction:
                if kp in absent_trg:
                    absent_tp += 1

            absent_precision[topk] = absent_tp / len(absent_prediction) if len(absent_prediction) > 0 else 0
            absent_recall[topk]    = absent_tp / len(absent_trg) if len(absent_trg) > 0 else 0

            if topk in total_present_precision:
                total_present_precision[topk] += present_precision[topk]
            else:
                total_present_precision[topk] = present_precision[topk]

            if topk in total_present_recall:
                total_present_recall[topk] += present_recall[topk]
            else:
                total_present_recall[topk] = present_recall[topk]

            if topk in total_absent_precision:
                total_absent_precision[topk] += absent_precision[topk]
            else:
                total_absent_precision[topk] = absent_precision[topk]

            if topk in total_absent_recall:
                total_absent_recall[topk] += absent_recall[topk]
            else:
                total_absent_recall[topk] = absent_recall[topk]

        total_data += 1
        i += 1

    return {"total_data": total_data,
            "total_present_precision": total_present_precision,
            "total_present_recall": total_present_recall,
            "total_absent_recall": total_absent_recall,
            "total_absent_precision": total_absent_precision}
