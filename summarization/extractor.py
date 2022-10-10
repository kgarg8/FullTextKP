import numpy as np, math, torch, torch.nn as nn, random, time, pdb
from pprint import pprint
from collections import Counter
from utils import evaluate_rouge
from bert_model import BertEdgeScorer, BertConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PacSumExtractor:

    def __init__(self, beta = 3, lambda1 = -0.2, lambda2 = -0.2):
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def extract_summary(self, data_iterator):
        raise NotImplementedError

    def _calculate_similarity_matrix(self, *inputs):
        raise NotImplementedError

    def _select_tops(self, edge_scores, beta, lambda1, lambda2):
        min_score = edge_scores.min()
        max_score = edge_scores.max()
        edge_threshold = min_score + beta * (max_score - min_score)
        new_edge_scores = edge_scores - edge_threshold
        forward_scores, backward_scores, _ = self._compute_scores(new_edge_scores, 0)
        forward_scores = 0 - forward_scores

        paired_scores = []
        for node in range(len(forward_scores)):
            paired_scores.append([node,  lambda1 * forward_scores[node] + lambda2 * backward_scores[node]])

        random.shuffle(paired_scores)   #shuffle to avoid any possible bias
        paired_scores.sort(key = lambda x: x[1], reverse = True)
        extracted = [item[0] for item in paired_scores]

        return extracted        # return re-ranked indices

    def _compute_scores(self, similarity_matrix, edge_threshold):

        forward_scores = [0 for i in range(len(similarity_matrix))]
        backward_scores = [0 for i in range(len(similarity_matrix))]
        edges = []
        for i in range(len(similarity_matrix)):
            for j in range(i+1, len(similarity_matrix[i])):
                edge_score = similarity_matrix[i][j]
                if edge_score > edge_threshold:
                    forward_scores[j] += edge_score
                    backward_scores[i] += edge_score
                    edges.append((i,j,edge_score))

        return np.asarray(forward_scores), np.asarray(backward_scores), edges

class PacSumExtractorWithBert(PacSumExtractor):

    def __init__(self, bert_model_file, bert_config_file, beta = 3, lambda1 = -0.2, lambda2 = -0.2):

        super(PacSumExtractorWithBert, self).__init__(beta, lambda1, lambda2)
        self.model = self._load_edge_model(bert_model_file, bert_config_file)

    def _calculate_similarity_matrix(self,  x, t, w, x_c, t_c, w_c, pair_indice):
        # doc: a list of sequences, each sequence is a list of words
        # x: token_ids, t: segment_ids, w: mask for every pair of sentences [#pairs = n*n-1] in the article. _c: co-pair document/ article

        def pairdown(scores, pair_indice, length):
            #1 for self score
            out_matrix = np.ones((length, length))
            for pair in pair_indice:
                # e.g. pair = ((0, 1), 2)
                out_matrix[pair[0][0]][pair[0][1]] = scores[pair[1]] # e.g. out_matrix[0][1] = 2; out_matrix[1][0] = 2
                out_matrix[pair[0][1]][pair[0][0]] = scores[pair[1]]

            return out_matrix

        scores = self._generate_score(x, t, w, x_c, t_c, w_c)
        doc_len = int(math.sqrt(len(x)*2)) + 1
        similarity_matrix = pairdown(scores, pair_indice, doc_len)

        return similarity_matrix

    def _generate_score(self, x, t, w, x_c, t_c, w_c):

        global device
        scores = torch.zeros(len(x)).to(device)
        step = 20
        for i in range(0,len(x),step):

            batch_x   = x[i:i+step]
            batch_t   = t[i:i+step]
            batch_w   = w[i:i+step]
            batch_x_c = x_c[i:i+step]
            batch_t_c = t_c[i:i+step]
            batch_w_c = w_c[i:i+step]

            inputs                   = tuple(t.to(device) for t in (batch_x, batch_t, batch_w, batch_x_c, batch_t_c, batch_w_c))
            batch_scores, batch_pros = self.model(*inputs)
            scores[i:i+step]         = batch_scores.detach()

        return scores

    def _load_edge_model(self, bert_model_file, bert_config_file):

        global device
        bert_config = BertConfig.from_json_file(bert_config_file)
        model = BertEdgeScorer(bert_config)
        model_states = torch.load(bert_model_file, map_location=torch.device(device))
        print(device)
        model.bert.load_state_dict(model_states)

        model = model.to(device)
        model.eval()
        return model

    def extract_summary(self, data_iterator):

        objs = {key: [] for key in ['src', 'src_vec', 'trg', 'trg_vec']}        # dictionary to finally dump
        article_count = 0
        for item in data_iterator:
            
            obj, inputs = item
            article_raw = obj['src']
            edge_scores = self._calculate_similarity_matrix(*inputs)
            ids = self._select_tops(edge_scores, beta=self.beta, lambda1=self.lambda1, lambda2=self.lambda2)

            # compute new src: <title> <sep> <abstract> <sep> <summary of body of document>
            indices = [i for i, x in enumerate(article_raw) if x == '<sep>']    # find all occurences of <sep>
            indices.append(len(article_raw))                                    # add end boundary so as to get the last sentence
            sublists = []
            title_abstract = article_raw[:indices[1]]
            for i in range(1, len(indices) - 1):                                # start from second <sep> to leave out title & abstract
                sublists.append(article_raw[indices[i]:indices[i+1]])
            summary = [sublists[i] for i in ids]
            summary = [item for sublist in summary for item in sublist]         # flatten lists of list in a list
            obj['src'] = title_abstract + summary
            
            # similarly compute summary_vec from src_vec
            subvecs = []
            title_abstract_vec = obj['src_vec'][:indices[1]]
            for i in range(1, len(indices) - 1):
                subvecs.append(obj['src_vec'][indices[i]:indices[i+1]])
            summary_vec = [subvecs[i] for i in ids]
            summary_vec = [item for sublist in summary_vec for item in sublist]
            obj['src_vec'] = title_abstract_vec + summary_vec
            
            # truncate at fixed length
            SRC_MAX_LEN = 800
            if len(obj['src']) > SRC_MAX_LEN:
                obj['src']     = obj['src'][:800]
                obj['src_vec'] = obj['src_vec'][:800]

            # append to dictionary
            objs['src'].append(obj['src'])
            objs['trg'].append(obj['trg'])
            objs['src_vec'].append(obj['src_vec'])
            objs['trg_vec'].append(obj['trg_vec'])

            # display
            article_count += 1
            if article_count % 500  == 0:
                print('\n\nTITLE & ABSTRACT')
                pprint(' '.join(title_abstract).split(' <sep> '), width=150)
                print('\n\n BODY:')
                pprint(' '.join(article_raw[indices[1]:]).split(' <sep> '), width=150)
                print('\n\nSUMMARY:')
                pprint(' '.join(summary).split(' <sep> '), width=150)
                print('\n\nRANKING:')
                print(ids)
                print('\n\nDUMP:')
                print('\n\nSRC: ' + ' '.join(obj['src']))
                print('\n\nSRC_VEC: ', obj['src_vec'])
                print('\n\nTRG: ' + ' '.join(obj['trg']))
                print('\n\nTRG_VEC: ', obj['trg_vec'])
                print('\n\nARTICLES PROCESSED: ', article_count)
        return objs

class PacSumExtractorWithTfIdf(PacSumExtractor):

    def __init__(self, beta = 3, lambda1 = -0.2, lambda2 = -0.2):
        super(PacSumExtractorWithTfIdf, self).__init__(beta, lambda1, lambda2)

    def _calculate_similarity_matrix(self, doc):

        idf_score = self._calculate_idf_scores(doc)
        tf_scores = [Counter(sentence) for sentence in doc]
        length = len(doc)
        similarity_matrix = np.zeros([length] * 2)

        for i in range(length):
            for j in range(i, length):
                similarity = self._idf_modified_dot(tf_scores, i, j, idf_score)
                if similarity:
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        return similarity_matrix

    def _idf_modified_dot(self, tf_scores, i, j, idf_score):

        if i == j:      return 1

        tf_i, tf_j = tf_scores[i], tf_scores[j]
        words_i, words_j = set(tf_i.keys()), set(tf_j.keys())

        score = 0
        for word in words_i & words_j:
            idf = idf_score[word]
            score += tf_i[word] * tf_j[word] * idf ** 2
        return score

    def _calculate_idf_scores(self, doc):

       doc_number_total = 0.
       df = {}
       for i, sen in enumerate(doc):
           tf = Counter(sen)
           for word in tf.keys():
               if word not in df:
                   df[word] = 0
               df[word] += 1
           doc_number_total += 1

       idf_score = {}
       for word, freq in df.items():
           idf_score[word] = math.log(doc_number_total - freq + 0.5) - math.log(freq + 0.5)

       return idf_score

    def extract_summary(self, data_iterator):

        objs = {key: [] for key in ['src', 'src_vec', 'trg', 'trg_vec']}     # dictionary to finally dump

        article_count = 0
        for item in data_iterator:
            obj, inputs = item
            edge_scores = self._calculate_similarity_matrix(inputs)
            ids = self._select_tops(edge_scores, beta=self.beta, lambda1=self.lambda1, lambda2=self.lambda2)

            # compute new src: <title> <sep> <abstract> <sep> <summary of body of document>
            article_raw = obj['src']
            indices = [i for i, x in enumerate(article_raw) if x == '<sep>'] # find all occurences of <sep>
            indices.append(len(article_raw))                                 # add end boundary so as to get the last sentence

            sublists = []
            title_abstract = article_raw[:indices[1]]
            for i in range(1, len(indices) - 1):                             # start from second <sep> to leave out title & abstract
                sublists.append(article_raw[indices[i]:indices[i+1]])

            summary = [sublists[i] for i in ids]
            summary = [item for sublist in summary for item in sublist]      # flatten lists of list in a list
            obj['src'] = title_abstract + summary

            # similarly compute summary_vec from src_vec
            subvecs = []
            title_abstract_vec = obj['src_vec'][:indices[1]]
            for i in range(1, len(indices) - 1):
                subvecs.append(obj['src_vec'][indices[i]:indices[i+1]])
            summary_vec = [subvecs[i] for i in ids]
            summary_vec = [item for sublist in summary_vec for item in sublist]
            obj['src_vec'] = title_abstract_vec + summary_vec

            # truncate at fixed length
            SRC_MAX_LEN = 800
            if len(obj['src']) > SRC_MAX_LEN:
                obj['src']     = obj['src'][:800]
                obj['src_vec'] = obj['src_vec'][:800]

            # append to dictionary
            objs['src'].append(obj['src'])
            objs['trg'].append(obj['trg'])
            objs['src_vec'].append(obj['src_vec'])
            objs['trg_vec'].append(obj['trg_vec'])

            # display
            article_count += 1
            if article_count % 500  == 0:
                print('\n\nTITLE & ABSTRACT')
                pprint(' '.join(title_abstract).split(' <sep> '), width=150)
                print('\n\n BODY:')
                pprint(' '.join(article_raw[indices[1]:]).split(' <sep> '), width=150)
                print('\n\nSUMMARY:')
                pprint(' '.join(summary).split(' <sep> '), width=150)
                print('\n\nRANKING:')
                print(ids)
                print('\n\nDUMP:')
                print('\n\nSRC: ' + ' '.join(obj['src']))
                print('\n\nSRC_VEC: ', obj['src_vec'])
                print('\n\nTRG: ' + ' '.join(obj['trg']))
                print('\n\nTRG_VEC: ', obj['trg_vec'])
                print('\n\nARTICLES PROCESSED: ', article_count)
        return objs
