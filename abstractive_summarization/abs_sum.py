import json, torch
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer
from tqdm import tqdm

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer  = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")
model      = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-arxiv").to(device)
batch_size = 12

def process(file):
    
    # provide path to extractive summarization dataset
    if file == 'dev' or file == 'test':
        filename = '../processed_data/ACM_{}_ACM.jsonl'.format(file)
    else: # train
        filename = '../processed_data/ACM_train.jsonl'
    
    data = []
    for line in open(filename, 'r'):
        data.append(json.loads(line))

    title_abstracts, abs_summary, targets = [], [], []

    for i in tqdm(range(0, len(data), batch_size)):
        inputs = []

        # create batches
        for j in range(batch_size):
            if i+j >= len(data): break
            src = ' '.join(data[i+j]['src'])
            src = src.split(' <sep> ', 2)
            t_a = src[0] + ' <sep> ' + src[1]
            body = src[2]
            inputs.append(body) # since we want to summarize only bodies
            targets.append(' '.join(data[i+j]['trg']))
            title_abstracts.append(t_a)

        # generate summaries
        tok_inputs  = tokenizer(inputs, return_tensors='pt', truncation=True, padding=True).to(device)
        predictions = model.generate(**tok_inputs)
        predictions = tokenizer.batch_decode(predictions)
        
        # post-process predictions
        preds = []
        for pred in predictions:
            pred = pred[4:] # to skip <s> token
            if pred.find('</s>') != -1:
                pred = pred[:pred.index('</s>')] # to remove </s> and <pad>
            pred = pred.replace('<n>', '')
            preds.append(pred)
        
        abs_summary.extend(preds)

    # append t_a and abs_summary
    data_dict = {}
    srcs = []
    for i in range(len(title_abstracts)):
        srcs.append(title_abstracts[i] + ' <sep> ' + abs_summary[i])

    # dump dict
    data_dict['src'] = srcs
    data_dict['trg'] = targets

    with open('abs_summary_stage1_{}.json'.format(file), 'w') as fp:
        json.dump(data_dict, fp)

process('train')
process('dev')
process('test')