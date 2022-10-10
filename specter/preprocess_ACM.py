# preprocess for specter format
# articles - title, abstract. Articles retrieved for train, dev, test sets.
# sentences - title = '', abstract = sentence. Sentences retrieved from title and abstract of only training data.
import json, jsonlines, time, pdb
from pathlib import Path
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from pprint import pprint

def processSentences(path, file):
    sentences_dict = {}
    articles_cnt = sent_cnt = 0

    for line in open('../processed_data/' + file, 'r'):
        data        = json.loads(line)
        src         = data['src']
        src_vec     = data['src_vec']
        sep_indices = [i for i, x in enumerate(src) if x == "<sep>"]

        text        = src[:sep_indices[1]]                       # title & abstract
        indices     = [i for i, x in enumerate(text) if x == "."]
        indices     = [sep_indices[0]] + indices        # append first <sep> index, where abstract starts
        if text[-1] != '.':    indices.append(len(text)) # append end of abstract, either . or <sep>
        
        title         = ' '.join(src[:sep_indices[0]])
        title_vec     =      src_vec[:sep_indices[0]]
        sentences     = [title] + [' '.join(src[indices[i]+1:indices[i+1]+1]) for i in range(0, len(indices)-1)]
        sentences_vec = [title_vec]  +  [src_vec[indices[i]+1:indices[i+1]+1] for i in range(0, len(indices)-1)]

        if articles_cnt % 5000 == 0:
            pprint(sentences, width=150)
            print('\n Articles processed: ', articles_cnt)
        
        # assertion
        for i in range(len(sentences)):
            sent = sentences[i].split(' ')
            sent = [item for item in sent if item != '']
             
            try:    assert len(sent) == len(sentences_vec[i])
            except: pdb.set_trace()

        for i in range(len(sentences)):
            sentences_dict[sent_cnt] = {'title' : '', 'abstract': sentences[i], 'src_vec': sentences_vec[i]}
            sent_cnt += 1

        articles_cnt+=1

    with open(path + 'sentences_' + file, 'w') as fp:
        json.dump(sentences_dict, fp)

def processArticles(path, file):

    articles_dict = {}
    articles_cnt = 0
    for line in open('../processed_data/' + file, 'r'):
        data    = json.loads(line)
        src     = data['src']
        src_vec = data['src_vec']
        indices = [i for i, x in enumerate(src) if x == "<sep>"]
        indices.append(len(src))
        
        title    = ' '.join(src[:indices[0]])
        abstract = ' '.join(src[indices[0]+1:indices[1]])
        articles_dict[articles_cnt] = {'title' : title, 'abstract' : abstract}
        articles_cnt += 1

        if articles_cnt % 5000 == 0:
            print('\n\nTitle: ', title)
            print('\n\nAbstract:', abstract)
            print('\n Articles processed: ', articles_cnt)

    with open(path + 'articles_' + file, 'w') as fp:
        json.dump(articles_dict, fp)

tic = time.perf_counter()

path = 'specter_format/'
Path(path).mkdir(parents=True, exist_ok=True)
processArticles (path, 'ACM_train.jsonl'   )
processArticles (path, 'ACM_dev_ACM.jsonl' )
processArticles (path, 'ACM_test_ACM.jsonl')
processSentences(path, 'ACM_train.jsonl'   )

toc = time.perf_counter()
print(f'Preprocessing for specter done in {toc - tic:0.4f} seconds')