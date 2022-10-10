# For Full Dataset
# Stage1 - abstract <sep> title <sep> collection of all sentences along with tags (C/NC)
import xml.etree.cElementTree as ElementTree, nltk, math, re, string, random, numpy as np, pickle, os, fnmatch, traceback, time, pprint, Levenshtein, pdb
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from functools import reduce
from os import fspath
from pathlib import Path
from preprocess_tools.process_utils import jsonl_save
from difflib import SequenceMatcher
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

random.seed(137)

class ProcessXML:

    def __init__(self):
        self.articles = []
        self.articles_count = self.total_cite_sent = self.total_sent = self.intro_count = 0

    def processxml_file(self, file):
        
        tree            = ElementTree.parse(file)
        root            = tree.getroot()
        reference_count = title_count = abstract_count = keywords_count = fulltext_count = valid_articles = 0

        # https://stackoverflow.com/questions/14946109/how-to-remove-escape-sequence-like-xe2-or-x0c-in-python
        remove_escape_x       = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]')
        remove_roman_numerals = re.compile(r'\([ivmcldx]+\)')
        remove_escape_symbols = re.compile(r'[\n\t]')
        remove_html           = re.compile(r'<[^>]*>')
        remove_urls           = re.compile(r'https?://(www\.)?\w+\.\w+')
        remove_emails         = re.compile(r'[\w.]+@[\w.]+')
        remove_num            = re.compile(r'\d+')
        citation_regex        = re.compile(r'\[\d[\d, ]*\]|\bet al|\[[A-Z][\w\s.]+\d{4}\]') # e.g. [2][4][3,5][Steiner and Ouni 2012][Musti et al. 2011] Walsh et al
        
        punkt_param              = PunktParameters()
        abbreviation             = ['al', 'i.e', 'e.g', 'etc', 'cf'] # et al., i.e., e.g., etc., cf.
        punkt_param.abbrev_types = set(abbreviation)
        tokenizer                = PunktSentenceTokenizer(punkt_param)

        article_node = root.findall('content/section/article_rec')
        if not article_node:
            article_node = root.findall('content/article_rec')
        if not article_node:    return
        for item in article_node:
            flag = intro_flag = cite_sent = total_sent = 0
            self.articles_count += 1
            article             = {}
            title               = ''
            for child in item:
                # title = append title & subtitle
                if child.tag == 'title':
                    title       = child.text.lower()
                    title       = re.sub(remove_html, '', title)
                    title       = re.sub(remove_num, '<digit>', title)
                    flag        += 1
                    title_count += 1

                elif child.tag == 'subtitle':
                    title += ' ' + child.text.lower()
                    title = re.sub(remove_html, '', title)
                    title = re.sub(remove_num, '<digit>', title)

                elif child.tag == 'abstract':
                    abstract       = child[0].text.lower()
                    abstract       = re.sub(remove_html, '', abstract)
                    abstract       = re.sub(remove_num, '<digit>', abstract)
                    flag           += 1
                    abstract_count += 1

                elif child.tag == 'keywords':
                    keywords = []
                    for grand_child in child:
                        if grand_child.tag == 'kw':
                            keyword = grand_child.text.lower()
                            keyword = re.sub(remove_num, '', keyword) # remove numbers completely from keywords
                            keywords.append(keyword)
                    keywords_count += 1
                    flag += 1

                elif child.tag == 'fulltext':
                    body = child.find('ft_body').text
                    pattern = re.compile(r'REFERENCES?')
                    match = pattern.search(body)
                    if match:  # Keep only papers with REFERENCES section
                        # Trim REFERENCES block, no citations there
                        body = body[:match.start()]
                        body = re.sub(remove_roman_numerals, '<digit>', body)
                        body = re.sub(remove_html, '', body)
                        body = re.sub(remove_escape_symbols, '', body)
                        body = re.sub(remove_escape_x, '', body)
                        body = re.sub(remove_emails, '', body)
                        body = re.sub(remove_urls, '', body)
                        
                        # Remove the sentence 'Permission... rst page.'
                        l1 = body.find('Permission')
                        l2 = body.find('rst page.')
                        if l1!= -1 and l2!= -1:
                            body = body[:l1] + body[l2+9:]
                        
                        # Remove the sentence 'To copy otherwise... a fee.'
                        l1 = body.find('To copy otherwise')
                        l2 = body.find('a fee.')
                        if l1!= -1 and l2!= -1:
                            body = body[:l1] + body[l2+6:]

                        pattern = re.compile('1. ?Introduction', re.IGNORECASE) # Match first occurence of introduction and trim everything before that
                        match   = pattern.search(body)
                        if match:       
                            body       = body[match.start():]
                            intro_flag = 1
                        else:
                            # remove title from full text
                            ratio = Levenshtein.ratio(body[:len(title)+1].lower(), title)
                            if ratio > 0.85: # 85% similarity
                                body = body[len(title)+1:]

                            # remove abstract from full text
                            a_idx = body.lower().find('abstract')
                            if a_idx != -1:
                                a_idx              += 9 # increment by 8 characters of abstract
                                extracted_abstract = body[a_idx: a_idx + len(abstract)].lower()
                                ratio             = Levenshtein.ratio(extracted_abstract, abstract)
                                if ratio > 0.75:
                                    body = body[a_idx + len(abstract):]
                    
                            # remove keywords from full text
                            k_idx = body.lower().find('keywords')
                            if k_idx != -1:
                                k_idx += 9
                                keywords_str       = ', '.join(keywords)
                                keywords_sorted    = ', '.join(sorted(keywords))
                                extracted_keywords = ', '.join(sorted(body[k_idx: k_idx + len(keywords_str)].lower().strip().split(', ')))
                                ratio = Levenshtein.ratio(extracted_keywords, keywords_sorted)
                                if ratio > 0.75:
                                    body = body[k_idx + len(keywords_str):]

                        segmented_body = tokenizer.tokenize(body)
                        
                        # keep only sentences > 5 words
                        segmented_body = [sent for sent in segmented_body if len(sent.split(' ')) > 5]

                        # remove citation strings & save citation indices
                        citation_sentence_idxs = []
                        for idx, sentence in enumerate(segmented_body):
                            if re.search(citation_regex, sentence):
                                segmented_body[idx] = re.sub(citation_regex, '', sentence)
                                citation_sentence_idxs.append(idx)
                        
                        # replace all numbers with <digit> after collecting citation sentences
                        for i in range(len(segmented_body)):
                            sentence          = re.sub(remove_num, '<digit>', segmented_body[i])
                            segmented_body[i] = sentence.rstrip(' .').lower() # Lower text here (tokenizer works better in sentence case)

                        citation_context = [segmented_body[i] for i in citation_sentence_idxs]
                        cite_sent        = len(citation_context)
                        citation_context = ' <sep> '.join(citation_context) # join list of sentences to form a string

                        total_sent = len(segmented_body)
                        full_text = ' <sep> '.join(segmented_body)

                        tag = ['C' if i in citation_sentence_idxs else 'NC' for i in range(total_sent)]
                        
                        fulltext_count += 1
                        flag += 1

                elif child.tag == 'references':
                    references      = []
                    ref_ids         = child.findall('ref/ref_seq_no')
                    ref_texts       = child.findall('ref/ref_text')
                    reference_count += 1
                    for ref_ids, ref_text in zip(ref_ids, ref_texts):
                        reference = ref_text.text
                        reference = re.sub(remove_roman_numerals, '<digit>', reference)
                        reference = re.sub(remove_html, '', reference)
                        reference = re.sub(remove_urls, '', reference)
                        reference = re.sub(remove_num, '<digit>', reference)
                        references.append(reference)
                    flag += 1

            if flag == 5 and cite_sent > 0:
                if intro_flag == 1:     self.intro_count += 1

                valid_articles              += 1
                article['title']            = title
                article['abstract']         = abstract
                article['keywords']         = keywords
                article['citation_context'] = citation_context
                article['full_text']        = full_text
                article['references']       = references
                article['tag']              = tag
                self.total_cite_sent        += cite_sent
                self.total_sent             += total_sent
                self.articles.append(article)

    def display(self):
        sample = random.sample(self.articles, 1)
        for i in range(len(sample)):
            print("\nTITLE: ", end="")
            print(sample[i]['title'])
            print("\nABSTRACT: ", end="")
            print(sample[i]['abstract'])
            print("\nKEYPHRASES: ", end="")
            print(sample[i]['keywords'])
            print("\nCITATION CONTEXT:\n")
            pprint.pprint(sample[i]['citation_context'].split(' <sep> '), width=150)
            print("\nFULL TEXT:\n")
            pprint.pprint(sample[i]['full_text'].split(' <sep> '), width=150)
            print("\nTags: ", end="")
            print(sample[i]['tag'])

vocab2count = {}
tokenizer = nltk.tokenize.MWETokenizer(separator='') # separator is inserted between multiple words when combining
tokenizer.add_mwe(('<','digit','>')) # strings to combine - <, digit, >
tokenizer.add_mwe(('<', 'sep', '>'))
tokenizer.add_mwe(('<', 'eos', '>'))

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

def process_data(processed_articles, idx1, idx2, train=True, update_vocab=True):
    def word_count(text):
        return sum([len(element.split(' ')) for element in text])

    srcs, trgs, tags = [], [], []
    for article in processed_articles[idx1: idx2]:
        
        src = article['title'] + ' <sep> ' + article['abstract'] + ' <sep> ' + article['full_text']
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

        tags.append(article['tag'])
        
        if update_vocab:
            for word in src:
                vocab2count[word] = vocab2count.get(word, 0) + 1
            for word in trg:
                vocab2count[word] = vocab2count.get(word, 0) + 1

    srcs_dict    = {}
    srcs_, trgs_, tags_ = [], [], []
    duplicate_count = full_duplicate_count = zero_trg_count = 0

    for src, trg, tag in zip(srcs, trgs, tags):
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

        if train and (len(trg) > 100):
            trg  = ' '.join(trg).split(';')
            trg_ = []
            i    = 0
            while len(trg_) <= 100:
                trg_ += trg[i].strip().split(' ') + [';']
                i    += 1
            trg_[-1] = '<eos>'
            trg      = trg_

        if len(trg) == 0:
            zero_trg_count += 1
            if train:
                flag = 1 # dont append if no keyphrases OR if duplicate article

        if flag == 0:
            srcs_.append(src)
            trgs_.append(trg)
            tags_.append(tag)

    srcs, trgs, tags = srcs_, trgs_, tags_

    print("Duplicate Count: {}, Full Duplicate Count: {}, Zero Trg Count: {}".format(duplicate_count, full_duplicate_count, zero_trg_count))
    assert len(srcs) == len(trgs) == len(tags)
    return srcs, trgs, tags

tic = time.perf_counter()
MAX_FILES = 500000
processed_articles = []
i = intro_count = cite_sent = total_sent = 0
for dirpath, dirs, files in os.walk('../../data/proceedings_small/'): # give path of the proceedings folder here
    # ignore hidden files and folders
    dirs[:] = [d for d in dirs if not d[0] == '.']
    files = [f for f in files if not f[0] == '.']

    for filename in fnmatch.filter(files, '*.xml'):
        print('Processing File: {}'.format(filename))
        try:
            xmlObject = ProcessXML()
            xmlObject.processxml_file(os.path.join(dirpath, filename))
            processed_articles += xmlObject.articles
            intro_count        += xmlObject.intro_count
            cite_sent          += xmlObject.total_cite_sent
            total_sent         += xmlObject.total_sent
            i                  += 1
            if i % 1 == 0:
                print('Processed File #{}: {}'.format(i, filename))
                xmlObject.display()
                print('Valid Articles Collected:', len(processed_articles))
                print('\n')
        except Exception as e:
            print()
            with open('stacktrace.txt', 'a') as f:
                f.write(filename + '\n')
                f.write(str(e) + '\n')
                f.write(traceback.format_exc() + '\n')
        print('Total processed articles: {}'.format(len(processed_articles)))
        print('Total Intro count: {}'.format(intro_count))
        print('Total Citation Sentences: {}'.format(cite_sent))
        print('Total Sentences: {}'.format(total_sent))
    if len(processed_articles) >= MAX_FILES:
        break

total_articles = len(processed_articles)
print('Total Valid Articles: ', total_articles)

idx1 = math.floor(total_articles * 0.8)
idx2 = math.floor(total_articles * 0.1)

dev_srcs, dev_trgs, dev_tags, test_srcs, test_trgs, test_tags = {}, {}, {}, {}, {}, {}
train_srcs, train_trgs, train_tags                   = process_data(processed_articles, 0, idx1, train=True, update_vocab=True)
dev_srcs['ACM'], dev_trgs['ACM'], dev_tags['ACM']    = process_data(processed_articles, idx1 + 1, idx1 + 1 + idx2, train=False, update_vocab=True)
test_srcs['ACM'], test_trgs['ACM'], test_tags['ACM'] = process_data(processed_articles, idx1 + 1 + idx2, total_articles, train=False, update_vocab=False)

toc = time.perf_counter()
print(f'Processed in {toc - tic:0.4f} seconds')
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

def text_vectorize(text):
    return [vocab2idx.get(word, vocab2idx['<unk>']) for word in text]

def vectorize_data(srcs, trgs, tags):
    data_dict            = {}
    srcs_vec             = [text_vectorize(src) for src in srcs]
    trgs_vec             = [text_vectorize(trg) for trg in trgs]
    data_dict['src']     = srcs
    data_dict['trg']     = trgs
    data_dict['src_vec'] = srcs_vec
    data_dict['trg_vec'] = trgs_vec
    data_dict['tag']     = tags
    return data_dict

Path('../processed_data/').mkdir(parents=True, exist_ok=True)

dev_data, test_data           = {}, {}
dev_save_path, test_save_path = {}, {}
dev_keys, test_keys           = ['ACM'], ['ACM']

train_save_path               = Path('../processed_data/ACM_train.jsonl')
metadata_save_path            = fspath(Path('../processed_data/ACM_metadata.pkl'))

for key in dev_keys:
    dev_save_path[key] = Path('../processed_data/ACM_dev_{}.jsonl'.format(key))
for key in test_keys:
    test_save_path[key] = Path('../processed_data/ACM_test_{}.jsonl'.format(key))

train_data = vectorize_data(train_srcs, train_trgs, train_tags)
jsonl_save(filepath=train_save_path, data_dict=train_data)

for key in dev_keys:
    dev_data[key] = vectorize_data(dev_srcs[key], dev_trgs[key], dev_tags[key])
    jsonl_save(filepath=dev_save_path[key], data_dict=dev_data[key])

for key in test_keys:
    test_data[key] = vectorize_data(test_srcs[key], test_trgs[key], test_tags[key])
    jsonl_save(filepath=test_save_path[key], data_dict=test_data[key])

metadata = {'dev_keys': dev_keys,
            'test_keys': test_keys,
            'vocab2idx': vocab2idx}

with open(metadata_save_path, 'wb') as outfile:
    pickle.dump(metadata, outfile)
