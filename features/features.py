import json
# Prepare for preprocessing

import pandas as pd
from tqdm import tqdm
from unidecode import unidecode
import jieba
from gensim import corpora, similarities, models
from gensim.summarization.bm25 import BM25
from gensim.models import Word2Vec
from gensim.corpora.mmcorpus import MmCorpus
from gensim.test.utils import datapath
import numpy as np
import math
import collections
from multiprocessing import cpu_count, Pool
import Levenshtein
import textdistance
from collections import defaultdict


class Feature():
    def __init__(self, data):
        """
        获取BM25打分、词向量、共现词统计等特征
        """
        self.data = data
        self.g_vec_model = ""
        self.query_ids = data['query_id']
        self.querys = data['query_text']
        self.querys_seg = self.querys.apply(self.get_seg_list)
        self.doc_ids = data['query_id']
        self.doc_text = data['doc_text']
        self.relevences = data['relevences']
        for doc_text in self.doc_text:
            pd.Series(doc_text).apply(self.get_seg_list).tolist()

        # self.doc_text_seg = pd.Series([pd.Series(doc_text).apply(self.get_seg_list).tolist() for doc_text in self.doc_text])
        self.doc_text_seg = self.doc_text.apply(self.get_seg_list)
        self.document_id_2_idx, self.query_map = self.init_dict()
        self.g_bm25_model = self.get_g_bm25_model(self.querys + self.doc_text)
        self.save_dict(self.querys_seg + self.doc_text_seg)
        # vec_model_path = './model/Word2Vec/GoogleNews-vectors-negative300.bin.gz'
        # g_vec_model = models.KeyedVectors.load_word2vec_format(vec_model_path, binary=True)
        self.g_dictionary = corpora.Dictionary.load('../data/model/Word2Vec/dictionary.dict')
        self.g_tfidf_model = models.TfidfModel.load("../data/model/Word2Vec/tfidf.model")
        self.g_index = similarities.SparseMatrixSimilarity.load('../data/model/Word2Vec/index.index')
        self.g_vec_model = Word2Vec.load('../data/model/Word2Vec/wiki.zh.text.model')

    def init_dict(self):
        document_map = dict()
        query_map = dict()
        # map doc_id to doc index in corpus
        document_id_2_idx = dict()
        doc_idx = 0
        for doc in zip(self.doc_ids, self.doc_text):
            document_map[doc[0]] = doc[1]
            document_id_2_idx[doc[0]] = doc_idx
            doc_idx += 1

        # query_export_pd = pd.read_csv(query_export_file)
        for query in zip(self.query_ids, self.querys):
            query_map[query[0]] = query[1]

        return document_id_2_idx, query_map

    def get_g_bm25_model(self, text_pool):
        """
        2d list
        :param text_pool:
        :return:
        """
        # text_pool = [line.split() for line in raw_text]
        word_freq = defaultdict(int)
        for line in text_pool:
            for word in line:
                word_freq[word] += 1
        text_pool = [[token for token in line if word_freq[token] > 1] for line in text_pool]
        g_bm25_model = BM25(text_pool)
        return g_bm25_model

    def save_dict(self, text_pool):
        dict_path = '../data/model/Word2Vec/'

        print('Initializing ...')
        dictionary = corpora.Dictionary(text_pool)
        corpus = [dictionary.doc2bow(line) for line in text_pool]
        tfidf_model = models.TfidfModel(corpus, dictionary=dictionary)
        corpus_tfidf = tfidf_model[corpus]
        print("Initialized")

        # Save dict and model
        dictionary.save(dict_path + 'dictionary.dict')
        tfidf_model.save(dict_path + 'tfidf.model')
        corpora.MmCorpus.serialize(dict_path + 'corpus.mm', corpus)
        num_features = len(dictionary.token2id.keys())
        # Similarities of sparse matrix
        index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=num_features)
        index.save(dict_path + 'index.index')

        print(num_features)

    def get_seg_list(self, text):
        """

        :param text:
        :return:
        """
        seg_list = []
        # if isinstance(text,list):
        #     seg_list = pd.DataFrame(text).apply(text)
        #
        # else:
        print(text)
        seg_list = list(jieba.cut_for_search(text))
        return seg_list

    def get_len(self, x):
        '''
            Length of tokens
        '''
        # x = x.split()
        return len(x)

    def get_token_cnt(self, x, y):
        '''
            Compute times of each token of y appeared in x
        '''
        # x = x.split()
        # y = y.split()
        num = 0
        for i in y:
            if i in x:
                num += 1
        return num

    def get_token_cnt_ratio(self, x, y):
        # x = x.split()
        return y / len(x)

    def get_jaccard_sim(self, x, y):
        '''
            Jaccard Similarity between x & y
        '''
        x = set(x)
        y = set(y)
        return float(len(x & y) / len(x | y))

    def get_mat_cos_sim(self, doc, corpus):
        '''
            Cosine Similarity between x & y
        '''
        # corpus = corpus.split(' ')
        # doc = doc.split(' ')

        corpus_vec = [self.g_dictionary.doc2bow(corpus)]
        vec = self.g_dictionary.doc2bow(doc)

        corpus_tfidf = self.g_tfidf_model[corpus_vec]
        vec_tfidf = self.g_tfidf_model[vec]

        num_features = len(self.g_dictionary.token2id.keys())
        mat_index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=num_features)
        sim = mat_index.get_similarities(vec_tfidf)

        return sim[0]

    def get_weight_counter_and_tf_idf(self, x, y):
        # x = x.split()
        # y = y.split()
        corups = x + y
        obj = dict(collections.Counter(corups))
        x_weight = []
        y_weight = []
        idfs = []
        for key in obj.keys():
            idf = 1
            w = obj[key]
            if key in x:
                idf += 1
                x_weight.append(w)
            else:
                x_weight.append(0)
            if key in y:
                idf += 1
                y_weight.append(w)
            else:
                y_weight.append(0)
            idfs.append(math.log(3.0 / idf) + 1)
        return [np.array(x_weight), np.array(y_weight), np.array(x_weight) * np.array(idfs),
                np.array(y_weight) * np.array(idfs), np.array(list(obj.keys()))]

    def get_manhattan_distance(self, x, y):
        '''
            Manhattan distance
        '''
        return np.linalg.norm(x - y, ord=1)

    def get_cos_sim(self, x, y):
        '''
            Cosine similarity between vectors
        '''
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    def get_euclidean_sim(self, x, y):
        '''
            Euclidean similarity between vectors
        '''
        return np.sqrt(np.sum(x - y) ** 2)

    def get_tfidf_sim(self, query, doc):
        '''
            TF-IDF
        '''
        weight = list(map(lambda x, y: self.get_weight_counter_and_tf_idf(x, y), tqdm(query), doc))
        x_weight_couner = []
        y_weight_couner = []
        x_weight_tfidf = []
        y_weight_tfidf = []
        words = []
        for i in weight:
            x_weight_couner.append(i[0])
            y_weight_couner.append(i[1])
            x_weight_tfidf.append(i[2])
            y_weight_tfidf.append(i[3])
            words.append(i[4])

        mht_sim_counter = list(map(lambda x, y: self.get_manhattan_distance(x, y), x_weight_couner, y_weight_couner))
        mht_sim_tfidf = list(map(lambda x, y: self.get_manhattan_distance(x, y), x_weight_tfidf, y_weight_tfidf))

        cos_sim_counter = list(map(lambda x, y: self.get_cos_sim(x, y), x_weight_couner, y_weight_couner))
        cos_sim_tfidf = list(map(lambda x, y: self.get_cos_sim(x, y), x_weight_tfidf, y_weight_tfidf))

        euclidean_sim_counter = list(map(lambda x, y: self.get_euclidean_sim(x, y), x_weight_couner, y_weight_couner))
        euclidean_sim_tfidf = list(map(lambda x, y: self.get_euclidean_sim(x, y), x_weight_tfidf, y_weight_tfidf))

        return mht_sim_counter, mht_sim_tfidf, cos_sim_counter, cos_sim_tfidf, euclidean_sim_counter, euclidean_sim_tfidf

    def get_word_vec(self, x):
        '''
            Word2Vec
        '''
        vec = []
        # for word in x.split():
        for word in x:
            if word in self.g_vec_model.wv:
                vec.append(self.g_vec_model.wv[word])
        if len(vec) == 0:
            return np.nan
        else:
            return np.mean(np.array(vec), axis=0)

    # def get_df_grams(self, train_sample, values, cols):
    #     def create_ngram_set(input_list, ngram_value):
    #         return set(zip(*[input_list[i:] for i in range(ngram_value)]))
    #
    #     def get_n_gram(df, values):
    #         train_query = df.values
    #         train_query = [[word for word in str(sen).replace("'", '').split(' ')] for sen in train_query]
    #         train_query_n = []
    #         for input_list in train_query:
    #             train_query_n_gram = set()
    #             for value in range(values, values + 1):
    #                 train_query_n_gram = train_query_n_gram | create_ngram_set(input_list, value)
    #             train_query_n.append(train_query_n_gram)
    #         return train_query_n
    #
    #     train_query = get_n_gram(train_sample[cols[0]], values)
    #     train_title = get_n_gram(train_sample[cols[1]], values)
    #     sim = list(map(lambda x, y: len(x) + len(y) - 2 * len(x & y), train_query, train_title))
    #     sim_number_rate = list(map(lambda x, y: len(x & y) / len(x) if len(x) != 0 else 0, train_query, train_title))
    #     return sim, sim_number_rate

    def get_n_grams(self, querys, docs, n):

        def create_ngram_set(input_list, ngram_value):
            return set(zip(*[input_list[i:] for i in range(ngram_value)]))

        def get_n_gram(words, values):

            train_query_n = []
            for input_list in words:
                train_query_n_gram = set()
                for value in range(values, values + 1):
                    train_query_n_gram = train_query_n_gram | create_ngram_set(input_list, value)
                train_query_n.append(train_query_n_gram)
            return train_query_n

        train_query = get_n_gram(querys, n)
        train_title = get_n_gram(docs, n)
        sim = list(map(lambda x, y: len(x) + len(y) - 2 * len(x & y), train_query, train_title))
        sim_number_rate = list(map(lambda x, y: len(x & y) / len(x) if len(x) != 0 else 0, train_query, train_title))
        return sim, sim_number_rate

    def get_token_matched_features(self, query, title):
        q_list = query.split()
        t_list = title.split()
        set_query = set(q_list)


        set_title = set(t_list)
        count_words = len(set_query.union(set_title))

        comwords = [word for word in t_list if word in q_list]
        comwords_set = set(comwords)
        unique_rate = len(comwords_set) / count_words

        same_word1 = [w for w in q_list if w in t_list]
        same_word2 = [w for w in t_list if w in q_list]
        same_len_rate = (len(same_word1) + len(same_word2)) / \
                        (len(q_list) + len(t_list))
        if len(comwords) > 0:
            com_index1 = len(comwords)
            same_word_q = com_index1 / len(q_list)
            same_word_t = com_index1 / len(t_list)

            for word in comwords_set:
                index_list = [i for i, x in enumerate(q_list) if x == word]
                com_index1 += sum(index_list)
            q_loc = com_index1 / (len(q_list) * len(comwords))
            com_index2 = len(comwords)
            for word in comwords_set:
                index_list = [i for i, x in enumerate(t_list) if x == word]
                com_index2 += sum(index_list)
            t_loc = com_index2 / (len(t_list) * len(comwords))

            same_w_set_q = len(comwords_set) / len(set_query)
            same_w_set_t = len(comwords_set) / len(set_title)
            word_set_rate = 2 * len(comwords_set) / \
                            (len(set_query) + len(set_title))

            com_set_query_index = len(comwords_set)
            for word in comwords_set:
                index_list = [i for i, x in enumerate(q_list) if x == word]
                if len(index_list) > 0:
                    com_set_query_index += index_list[0]
            loc_set_q = com_set_query_index / (len(q_list) * len(comwords_set))
            com_set_title_index = len(comwords_set)
            for word in comwords_set:
                index_list = [i for i, x in enumerate(t_list) if x == word]
                if len(index_list) > 0:
                    com_set_title_index += index_list[0]
            loc_set_t = com_set_title_index / (len(t_list) * len(comwords_set))
            set_rate = (len(comwords_set) / len(comwords))
        else:
            unique_rate, same_len_rate, same_word_q, same_word_t, q_loc, t_loc, same_w_set_q, same_w_set_t, word_set_rate, loc_set_q, loc_set_t, set_rate = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        return unique_rate, same_len_rate, same_word_q, same_word_t, q_loc, t_loc, same_w_set_q, same_w_set_t, word_set_rate, loc_set_q, loc_set_t, set_rate

    def get_substr_features(self, query, title):
        # q_list = query.split()
        q_list = query
        query_len = len(q_list)
        # t_list = title.split()
        t_list = title
        title_len = len(t_list)
        count1 = np.zeros((query_len + 1, title_len + 1))
        index = np.zeros((query_len + 1, title_len + 1))
        for i in range(1, query_len + 1):
            for j in range(1, title_len + 1):
                if q_list[i - 1] == t_list[j - 1]:
                    count1[i][j] = count1[i - 1][j - 1] + 1
                    index[i][j] = index[i - 1][j - 1] + j
                else:
                    count1[i][j] = 0
                    index[i][j] = 0
        max_count1 = count1.max()

        if max_count1 != 0:
            row = int(np.where(count1 == np.max(count1))[0][0])
            col = int(np.where(count1 == np.max(count1))[1][0])
            mean_pos = index[row][col] / (max_count1 * title_len)
            begin_loc = (col - max_count1 + 1) / title_len
            rows = np.where(count1 != 0.0)[0]
            cols = np.where(count1 != 0.0)[1]
            total_loc = 0
            for i in range(0, len(rows)):
                total_loc += index[rows[i]][cols[i]]
            density = total_loc / (query_len * title_len)
            rate_q_len = max_count1 / query_len
            rate_t_len = max_count1 / title_len
        else:
            begin_loc, mean_pos, total_loc, density, rate_q_len, rate_t_len = 0, 0, 0, 0, 0, 0
        return max_count1, begin_loc, mean_pos, total_loc, density, rate_q_len, rate_t_len

    def get_common_words(self, query, title):
        # query = set(query.split())
        # title = set(title.split())
        query = set(query)
        title = set(title)
        return len(query & title)

    def get_bm25_group(self, df):
        '''
            Build BM25 model for each query group
        '''
        df.columns = ['query_id', 'query_text', 'doc_text']
        df['query_id'] = df['query_id'].fillna('always_nan')
        query_id_group = df.groupby(['query_id'])
        bm_list = []
        for name, group in tqdm(query_id_group):
            group_corpus = group['doc_text'].values.tolist()
            group_corpus = [sentence.strip().split() for sentence in group_corpus]
            query = group['query_text'].values[0].strip().split()
            group_bm25_model = BM25(group_corpus)
            # group_average_idf = sum(map(lambda k: float(group_bm25_model.idf[k]), group_bm25_model.idf.keys())) / len(group_bm25_model.idf.keys())
            bm_score = group_bm25_model.get_scores(query)  # group_average_idf)
            bm_list.extend(bm_score)

        return bm_list

    def get_bm25_overall(self, doc_id, query_text):
        '''
            Compute BM25 with model over all documents
        '''
        score = self.g_bm25_model.get_score(query_text, self.document_id_2_idx[doc_id])  # g_average_idf
        return score

    def get_feature(self):
        result = {}
        feat_prefix = 'feat_'
        query_len = self.querys_seg.apply(self.get_len)
        doc_len = self.doc_text_seg.apply(self.get_len)

        jaccard_sim = list(map(self.get_jaccard_sim, self.querys_seg, self.doc_text_seg))
        edit_distance = list(
            map(lambda x, y: Levenshtein.distance(x, y) / (len(x) + 1), self.querys, self.doc_text))
        edit_jaro = list(
            map(lambda x, y: Levenshtein.jaro(x, y), self.querys, self.doc_text))
        edit_ratio = list(
            map(lambda x, y: Levenshtein.ratio(x, y), self.querys, self.doc_text))
        edit_jaro_winkler = list(
            map(lambda x, y: Levenshtein.jaro_winkler(x, y), self.querys, self.doc_text))
        hamming = list(
            map(lambda x, y: textdistance.Hamming(qval=None).normalized_distance(x, y), self.querys, self.doc_text))

        mht_sim, tf_mht_sim, cos_sim, tf_cos_sim, euc_sim, tf_euc_sim = self.get_tfidf_sim(self.querys_seg,
                                                                                           self.doc_text_seg)
        gram_2_sim, gram_2_sim_ratio = self.get_n_grams(self.querys_seg, self.doc_text_seg, 2)
        gram_3_sim, gram_3_sim_ratio = self.get_n_grams(self.querys_seg, self.doc_text_seg, 3)

        bm25_group = self.get_bm25_group(self.data[['query_id', 'query_text', 'doc_text']])
        bm25_overall = list(map(self.get_bm25_overall, self.doc_ids, self.querys_seg))
        mat_cos_sim = list(map(lambda x, y: self.get_mat_cos_sim(x, y), self.querys_seg, self.doc_text_seg))
        query_vec = self.querys_seg.apply(lambda x: self.get_word_vec(x))
        doc_vec = self.doc_text_seg.apply(lambda x: self.get_word_vec(x))
        cos_mean_word2vec = list(map(self.get_cos_sim, query_vec, doc_vec))
        euc_mean_word2vec = list(map(self.get_euclidean_sim, query_vec, doc_vec))
        mhd_mean_word2vec = list(map(self.get_manhattan_distance, query_vec, doc_vec))
        # 'query_vec': query_vec, 'doc_vec': doc_vec,
        result = {'query_id': self.query_ids, 'query_text': self.querys, 'doc_id': self.doc_ids, 'doc_text': self.doc_text,
                  'relevence': self.relevences,
                  'query_len': query_len, 'doc_len': doc_len, 'jaccard_sim': jaccard_sim,
                  'edit_distance': edit_distance, 'edit_jaro': edit_jaro, 'edit_ratio': edit_ratio,
                  'edit_jaro_winkler': edit_jaro_winkler, 'hamming': hamming,
                  'mht_sim': mht_sim, 'tf_mht_sim': tf_mht_sim, 'cos_sim': cos_sim, 'tf_cos_sim': tf_cos_sim,
                  'euc_sim': euc_sim, 'tf_euc_sim': tf_euc_sim, 'gram_2_sim': gram_2_sim,
                  'gram_2_sim_ratio': gram_2_sim_ratio,
                  'gram_3_sim': gram_3_sim, 'gram_3_sim_ratio': gram_3_sim_ratio,
                  'bm25_group': bm25_group, 'bm25_overall': bm25_overall, 'mat_cos_sim': mat_cos_sim,
                  'cos_mean_word2vec': cos_mean_word2vec,
                  'euc_mean_word2vec': euc_mean_word2vec, 'mhd_mean_word2vec': mhd_mean_word2vec}

        return pd.DataFrame(result)


if __name__ == '__main__':
    # query_dict = {'保险增员', ["保险增员", "保险增员话术", "保险增员话术之三",
    #                        "如何进行增员", "增员的技巧", "山东如何进行保险增员",
    #                        "保险增员有什么技巧吗", "看销售冠军如何进行保险增员"]}
    ids = [1, 2]
    querys = ["保险增员", "犹豫期"]
    docs = [["保险增员", "保险增员话术", "保险增员话术之三",
             "如何进行增员", "增员的技巧", "山东如何进行保险增员",
             "保险增员有什么技巧吗", "看销售冠军如何进行保险增员"],
            ["买保险有犹豫期吗", "保险的犹豫期有多久", "车险有犹豫期吗", "保险法犹豫期30天？",
             "什么是保险犹豫期？", "保单的犹豫期", "犹豫期退保什么意思", "犹豫期从什么时候开始计算"]]
    docs_id = [["111", "112", "113", "114", "115", "116", "117", "118"],
               ["211", "212", "213", "214", "215", "216", "217", "218"]]
    relevences = [[3, 2, 1, 3, 2, 3, 1, 0], [3, 3, 2, 3, 1, 2, 3, 0]]
    length = len(querys)
    # for i,query in enumerate(querys):
    #     for j,doc in enumerate(docs):
    query_1 = []
    id_1 = []
    query_2 = []
    id_2 = []
    for i, query in enumerate(querys * 8):
        if i % 2 == 0:
            query_1.append(query)
            id_1.append(ids[0])
        else:
            query_2.append(query)
            id_2.append(ids[1])
    new_querys = []
    new_ids = []
    new_querys = query_1 + query_2
    new_ids = id_1 + id_2

    train_data = pd.DataFrame({'query_id': new_ids, 'query_text': new_querys,
                               'doc_text': sum(docs, []), 'doc_id': sum(docs_id, []),
                               'relevences': sum(relevences, [])})

    data = pd.DataFrame({'query_id': ids, 'query_text': querys, 'doc_text': docs, 'doc_id': docs_id})
    data.to_csv('../data/test.csv', index=False)
    train_data.to_csv('../data/train.csv', index=False)
    feature = Feature(train_data)
    feature_map = feature.get_feature()
    print(feature_map)
