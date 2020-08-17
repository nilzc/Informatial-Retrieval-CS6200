from elasticsearch import Elasticsearch
from elasticsearch import helpers
from elasticsearch.client import IndicesClient
import json
import re
import math
import pandas as pd


def analyze_word(_key):
    result = indices_es.analyze(index="ap_files_stop_stem",
                                body={
                                    "analyzer": "stopped_stem",
                                    "text": _key
                                })
    result = [aq['token'] for aq in result['tokens']]
    return result


def compute_otf(tf, dl, avg_dl):
    return tf / (tf + 0.5 + 1.5 * (dl / avg_dl))


def compute_tfidf(tf, dl, avg_dl, D, df):
    return (tf / (tf + 0.5 + 1.5 * (dl / avg_dl))) * math.log(D / df)


def compute_bm(tf, dl, avg_dl, D, df):
    BM = math.log((D + 0.5) / (df + 0.5)) \
         * (tf + k1 * tf) / (tf + k1 * ((1 - b) + b * (dl / avg_dl)))
    return BM


def compute_lml(tf, dl, V):
    lml = math.log((tf + 1) / (dl + V))
    return lml


def compute_lmjm(tf, dl, V, ttf, _lambda):
    lmjm = math.log(_lambda * tf / dl + (1 - _lambda) * ttf / V)
    return lmjm


def get_top_1000(doc):
    doc_ids = sorted(doc, key=doc.get, reverse=True)[0:1000]
    query_doc = [{doc_id: doc[doc_id]} for doc_id in doc_ids]
    return query_doc


def add_score(id, doc, score):
    if id in doc:
        doc[id] += score
    else:
        doc[id] = score
    return doc


def write_score_file(file_name, query_doc, query_id, line_num):
    with open("./model_score/" + file_name, "a") as f:
        temp_item = query_doc[line_num - 1]
        temp_id = ''.join(temp_item)
        write_line = query_id + " " + "Q0" + " " + temp_id + " " + str(i) + " " + str(temp_item[temp_id]) + " Exp"
        f.writelines("%s\n" % write_line)


def read_parsed_file():
    with open("./parsed_file.json", "r") as f:
        files = json.load(f)
    return files


# read queries
def read_queries(file_name):
    queries = []
    queries_id = []
    with open("./" + file_name, "r") as f:
        for i in f.readlines():
            queries.append(re.findall("[A-Z|a-z].*[a-z]", i))
            queries_id.append(re.findall("^[0-9]+", i)[0])
    return queries, queries_id


# read term vectors from Elasticsearch
def read_term_vectors(result, ids, size):
    n_package = len(ids) // size
    if n_package > 0:
        for i in range(0, n_package):
            print(i)
            term_vecs = es.mtermvectors(index="ap_files_stop_stem",
                                        body={
                                            "ids": ids[size * i:size * (i + 1)],
                                            "parameters": {
                                                "fields": ["content"],
                                                "offsets": "false",
                                                "payloads": "false",
                                                "positions": "false",
                                                "term_statistics": "true",
                                                "field_statistics": "false"
                                            }
                                        })['docs']
            for term_vec in term_vecs:
                if len(term_vec['term_vectors']) == 0:
                    continue
                term_id = term_vec['_id']
                terms = term_vec['term_vectors']['content']['terms']
                key = [k for k in terms.keys()][0]
                exp = es.explain(index="ap_files_stop_stem", id=term_id,
                                 body={
                                     "query": {"term": {"content": key}}
                                 })
                dl = exp['explanation']['details'][0]['details'][2]['details'][3]['value']
                avg_dl = exp['explanation']['details'][0]['details'][2]['details'][4]['value']

                terms['_doc'] = {"dl": dl, "avg_dl": avg_dl}
                # add to the cache
                result[term_id] = terms
    else:
        i = -1
    term_vecs = es.mtermvectors(index="ap_files_stop_stem",
                                body={
                                    "ids": ids[size * (i + 1):],
                                    "parameters": {
                                        "fields": ["content"],
                                        "offsets": "false",
                                        "payloads": "false",
                                        "positions": "false",
                                        "term_statistics": "true",
                                        "field_statistics": "false"
                                    }
                                })['docs']
    for term_vec in term_vecs:
        if len(term_vec['term_vectors']) == 0:
            continue
        term_id = term_vec['_id']
        terms = term_vec['term_vectors']['content']['terms']
        key = [k for k in terms.keys()][0]
        exp = es.explain(index="ap_files_stop_stem", id=term_id,
                         body={
                             "query": {"term": {"content": key}}
                         })
        dl = exp['explanation']['details'][0]['details'][2]['details'][3]['value']
        avg_dl = exp['explanation']['details'][0]['details'][2]['details'][4]['value']

        terms['_doc'] = {"dl": dl, "avg_dl": avg_dl}
        # add to the cache
        result[term_id] = terms
    return result

# PREPARATION
#############################################################################################


# read parsed docs
files = read_parsed_file()
es = Elasticsearch(timeout=60)
indices_es = IndicesClient(es)
# add index template
es.indices.put_template(name="my_template",
                        body={
                            "index_patterns": "enter_index_name_pattern",
                            "settings": {
                                "number_of_replicas": 1,
                                "analysis": {
                                    "filter": {
                                        "english_stop": {
                                            "type": "stop",
                                            "stopwords_path": "stoplist.txt",
                                        },
                                        "my_snow": {
                                            "type": "snowball",
                                            "language": "English"
                                        }
                                    },
                                    "analyzer": {
                                        # custom analyzer "stopped"
                                        "stopped_stem": {
                                            "type": "custom",
                                            "tokenizer": "standard",
                                            "filter": [
                                                "lowercase",
                                                # custom filter "english_stop"
                                                "english_stop",
                                                "my_snow"
                                            ]
                                        }
                                    }
                                }
                            },
                            "mappings": {
                                "properties": {
                                    # filed name is "content"
                                    "content": {
                                        "type": "text",
                                        "fielddata": True,
                                        "analyzer": "stopped_stem",
                                        "index_options": "positions"
                                    },
                                    # to use size
                                    "_size": {
                                        "enabled": "true"
                                    }
                                }
                            }
                        })
# create a new index
es.indices.create(index="ap_files_stop_stem")
# add data using BULK API
actions = [
    {
        "_index": "ap_files_stop_stem",
        "_id": i,
        "_source": {
            "content": files[i]
        }
    }
    for i in files
]
helpers.bulk(es, actions=actions)

# read queries
query_file = "queries.txt"
queries, queries_id = read_queries(query_file)


# CONSTANTS AND NEEDED OBJECTS
#############################################################################################


k1 = 1.2
k2 = 100
b = 0.75
D = 84678
_lambda = 0.9
V = es.search(index="ap_files_stop_stem",
              body={
                  "aggs": {
                      "unique_terms": {
                          "cardinality": {
                              "field": "content",
                              # default is 3000, maximum is 40000
                              "precision_threshold": 40000
                          }
                      }
                  }
              })['aggregations']['unique_terms']['value']
total_words = es.termvectors(index="ap_files_stop_stem",
                             id='AP890902-0162',
                             body={
                                 "fields": ["content"],
                                 "offsets": "false",
                                 "payloads": "false",
                                 "positions": "false",
                                 "term_statistics": "false",
                                 "field_statistics": "true"
                             })['term_vectors']['content']['field_statistics']['sum_ttf']

query_rel_doc_ids = {}
terms_ttf = {}
term_vectors = {}
version = "hw1"

# GET TERM VECTORS & CALCULATE SCORES
#############################################################################################


def calculate_score(query_file, version):
    global query_rel_doc_ids
    global terms_ttf
    global term_vectors

    if len(query_rel_doc_ids) == 0:
        for idx, q in enumerate(query_file):
            query_id = queries_id[idx]
            target_ids = helpers.scan(es, index="ap_files_stop_stem",
                                      query={
                                          "query": {"match": {"content": q[0]}},
                                          "stored_fields": []
                                      })
            target_ids = [t['_id'] for t in target_ids]
            print(query_id, len(target_ids))
            query_rel_doc_ids[query_id] = target_ids

    # get ttf for all keywords in the queries
    if len(terms_ttf) == 0:
        for idx, q in enumerate(query_file):
            query_id = queries_id[idx]
            unanalyzed_q = re.split(" |-", q[0])
            print(idx + 1, query_id)
            for key in unanalyzed_q:
                if key in ['increas']:
                    key += 'e'
                analyzed_q = analyze_word(key)
                if len(analyzed_q) == 0:
                    continue
                else:
                    analyzed_q = analyzed_q[0]

                    key_search = es.search(index="ap_files_stop_stem",
                                           body={
                                               "size": 1,
                                               "query": {"match": {"content": key}},
                                               "stored_fields": []
                                           })['hits']['hits'][0]['_id']
                    key_term_vec = es.termvectors(index="ap_files_stop_stem", id=key_search,
                                       body={
                                           "fields": ["content"],
                                           "offsets": "false",
                                           "payloads": "false",
                                           "positions": "false",
                                           "term_statistics": "true",
                                           "field_statistics": "false"
                                       })['term_vectors']['content']['terms'][analyzed_q]
                    ttf = key_term_vec['ttf']
                    terms_ttf[analyzed_q] = ttf

    for idx, q in enumerate(query_file):
        query_id = queries_id[idx]
        unanalyzed_q = re.split(" |-", q[0])
        analyzed_q = analyze_word(q[0])
        target_ids = query_rel_doc_ids[query_id]
        # dict to store score
        doc_score_OTF = {}
        doc_score_TFIDF = {}
        doc_score_BM = {}
        doc_score_LM_L = {}
        doc_score_LM_JM = {}
        print(query_id, q)
        # fill term vectors
        missing_docs = []
        for doc_id in target_ids:
            if doc_id in term_vectors:
                term_vec = term_vectors[doc_id]
            else:
                missing_docs.append(doc_id)
        if len(missing_docs) != 0:
            term_vectors = read_term_vectors(term_vectors, missing_docs, 5000)
        # calculate score
        for doc_id in target_ids:
            target_terms = term_vectors[doc_id]
            for key in analyzed_q:
                if key in ['increa']:
                    key += 's'
                if key not in target_terms:
                    # language model
                    ttf = terms_ttf[key]
                    dl = target_terms['_doc']['dl']
                    LML = compute_lml(0, dl, V)
                    LMJM = compute_lmjm(0, dl, V, ttf, _lambda)
                    doc_score_LM_L = add_score(doc_id, doc_score_LM_L, LML)
                    doc_score_LM_JM = add_score(doc_id, doc_score_LM_JM, LMJM)
                    continue
                else:
                    key_term_vec = target_terms[key]
                    ttf = key_term_vec['ttf']
                    tf = key_term_vec['term_freq']
                    df = key_term_vec['doc_freq']
                    dl = target_terms['_doc']['dl']
                    avg_dl = target_terms['_doc']['avg_dl']

                    OTF = compute_otf(tf, dl, avg_dl)
                    TFIDF = compute_tfidf(tf, dl, avg_dl, D, df)
                    BM = compute_bm(tf, dl, avg_dl, D, df)
                    LML = compute_lml(tf, dl, V)
                    LMJM = compute_lmjm(tf, dl, V, ttf, _lambda)

                    doc_score_OTF = add_score(doc_id, doc_score_OTF, OTF)
                    doc_score_TFIDF = add_score(doc_id, doc_score_TFIDF, TFIDF)
                    doc_score_BM = add_score(doc_id, doc_score_BM, BM)
                    doc_score_LM_L = add_score(doc_id, doc_score_LM_L, LML)
                    doc_score_LM_JM = add_score(doc_id, doc_score_LM_JM, LMJM)
        # top 1000 docs
        query_doc_OTF = get_top_1000(doc_score_OTF)
        query_doc_TFIDF = get_top_1000(doc_score_TFIDF)
        query_doc_BM = get_top_1000(doc_score_BM)
        query_doc_LML = get_top_1000(doc_score_LM_L)
        query_doc_LMJM = get_top_1000(doc_score_LM_JM)
        # es-builtin
        q_match = es.search(index="ap_files_stop_stem",
                            body={
                                "size": 1000,
                                "query": {"match": {"content": q[0]}}
                            })
        query_doc_es = [{i['_id']: i['_score']} for i in q_match['hits']['hits']]

        for i in range(1, 1500):
            if i <= len(query_doc_OTF):
                write_score_file("OTF_" + version + ".txt", query_doc_OTF, query_id, i)
            if i <= len(query_doc_TFIDF):
                write_score_file("TFIDF_" + version + ".txt", query_doc_TFIDF, query_id, i)
            if i <= len(query_doc_BM):
                write_score_file("BM_" + version + ".txt", query_doc_BM, query_id, i)
            if i <= len(query_doc_es):
                write_score_file("ES_" + version + ".txt", query_doc_es, query_id, i)
            if i <= len(query_doc_LML):
                write_score_file("LML_" + version + ".txt", query_doc_LML, query_id, i)
            if i <= len(query_doc_LMJM):
                write_score_file("LMJM_" + str(_lambda) + "_" + version + ".txt", query_doc_LMJM, query_id, i)


calculate_score(queries, version)

# PSEUDO RELEVANCE FEEDBACK
#############################################################################################


# read interesting words
with open("./interesting_1.json", "r") as f:
    interesting_words_1 = json.load(f)
with open("./interesting_2.json", "r") as f:
    interesting_words_2 = json.load(f)
with open("./interesting_es.json", "r") as f:
    interesting_words_es = json.load(f)
with open("./interesting_diff.json", "r") as f:
    interesting_diff = json.load(f)

queries = []
queries_id = []
with open("./queries.txt", "r") as f:
    for i in f.readlines():
        queries.append(re.findall("[A-Z|a-z].*[A-Z|a-z]", i))
        queries_id.append(re.findall("^[0-9]+", i)[0])

# get the first k docs ids
for idx, q in enumerate(queries):
    query_id = queries_id[idx]
    unanalyzed_q = re.split(" |-", q[0])
    analyzed_q = analyze_word(q[0])
    # different model score
    doc_score_TFIDF = {}
    print(idx + 1, query_id, analyzed_q)
    with open("ap_query_term_vectors/" + query_id + "_ids.json", "r") as f:
        target_ids = f.read().split("\n")[:-1]
    for target_id in target_ids:
        target_terms = term_vectors[target_id]
        for key in analyzed_q:
            if key not in target_terms:
                continue
            else:
                key_term_vec = target_terms[key]
                ttf = key_term_vec['ttf']
                tf = key_term_vec['term_freq']
                df = key_term_vec['doc_freq']
                dl = target_terms['_doc']['dl']
                avg_dl = target_terms['_doc']['avg_dl']

                TFIDF = compute_tfidf(tf, dl, avg_dl, D, df)

                doc_score_TFIDF = add_score(target_id, doc_score_TFIDF, TFIDF)
    query_doc_TFIDF = get_top_1000(doc_score_TFIDF)
    # first k docs
    first_n = query_doc_TFIDF[:30]

# term vectors for first k docs
query_term_vectors = [term_vectors[t] for t in first_n]

# identify interesting words
# method 1: tf/dl - ttf/total
# method 2: cf/k - df/D
interesting_words_1 = {}
interesting_words_2 = {}
for idx, q in enumerate(queries):
    query_id = queries_id[idx]
    query_docs = query_term_vectors[idx * 30:idx * 30 + 30]
    total_terms = []
    # dl for the top 30 docs
    top_dl = 0
    for doc in query_docs:
        dl = doc['_doc']['dl']
        top_dl += dl
        avg_dl = doc['_doc']['avg_dl']
        for key in doc:
            if key == '_doc':
                continue
            key_term_vec = doc[key]
            term = key
            tf = key_term_vec['term_freq']
            ttf = key_term_vec["ttf"]
            df = key_term_vec["doc_freq"]
            total_terms.append([term, tf, ttf, df, dl, avg_dl])
    # method 1
    total_terms_df = pd.DataFrame(total_terms, columns=["term", "tf", "ttf", "df", "dl", "avg_dl"], dtype=float)
    method_1 = pd.DataFrame({'term': list(total_terms_df.groupby('term').sum().index),
                             'tf': list(total_terms_df.groupby('term').sum()['tf'])
                             })
    method_1 = pd.merge(left=method_1, right=total_terms_df.loc[:, ['term', 'ttf']].drop_duplicates(),
                        on='term', how='left')
    method_1['tf/dl'] = method_1['tf'] / top_dl
    method_1['ttf/total'] = method_1['ttf'] / total_words
    method_1['score'] = method_1['tf/dl'] - method_1['ttf/total']
    method_1.sort_values(by='score', ascending=False).head(50)
    interesting_words_1[query_id] = list(method_1.sort_values(by='score', ascending=False).head(30)['term'])
    # method 2
    total_appear = total_terms_df.groupby('term').count().sort_values(by='tf', ascending=False)['tf']
    total_appear = pd.DataFrame({'term': total_appear.index, 'cf': total_appear})
    total_appear.reset_index(drop=True, inplace=True)
    total_sig = pd.merge(left=total_appear, right=total_terms_df.loc[:, ['term', 'df']].drop_duplicates(), on='term',
                         how="inner")
    # total_sig['delta'] = total_sig['cf'] / (total_sig['df'] + 10000)
    total_sig['delta'] = total_sig['cf'] / 30 - total_sig['df'] / D
    interesting_words_2[query_id] = list(total_sig.sort_values(by='delta', ascending=False).head(30)['term'])

with open("./interesting_1.json", "w") as f:
    json.dump(interesting_words_1, f)
with open("./interesting_2.json", "w") as f:
    json.dump(interesting_words_2, f)

# get the different words from significant API
interesting_diff = {}
for q in interesting_words_es:
    words = interesting_words_es[q]
    compare_words = interesting_words_2[q]
    difference = []
    for word in words:
        if word not in compare_words:
            difference.append(word)
    interesting_diff[q] = difference
with open("./interesting_diff.json", "w") as f:
    json.dump(interesting_diff, f)

# RE-RUN PROGRAM
#############################################################################################


queries = []
queries_id = []
with open("./queries_short.txt", "r") as f:
    for i in f.readlines():
        queries.append(re.findall("[A-Z|a-z].*[A-Z|a-z]", i))
        queries_id.append(re.findall("^[0-9]+", i)[0])

# modify queries based on pseudo-relevance feedback
queries_m1 = []
queries_m2 = []
queries_es = []
for idx, q in enumerate(queries):
    query_id = queries_id[idx]
    words_origin = analyze_word(q[0])
    words_m1 = interesting_words_1[query_id]
    words_m2 = interesting_words_2[query_id]
    words_es = interesting_words_es[query_id]
    for key in words_origin:
        if key in words_es:
            words_es.remove(key)
        if key in words_m1:
            words_m1.remove(key)
        if key in words_m2:
            words_m2.remove(key)
    words_final_m1 = words_origin + words_m1[:5]
    words_final_m2 = words_origin + words_m2[:5]
    words_final_es = words_origin + words_es[:5]
    queries_m1.append([' '.join(words_final_m1)])
    queries_m2.append([' '.join(words_final_m2)])
    queries_es.append([' '.join(words_final_es)])

version = "short_modified_es"
for idx, q in enumerate(queries_es):
    query_id = queries_id[idx]
    unanalyzed_q = re.split(" |-", q[0])
    analyzed_q = analyze_word(q[0])
    # different model score
    doc_score_TFIDF = {}
    print(idx + 1, query_id, analyzed_q)
    target_ids = query_rel_doc_ids[query_id]
    for target_id in target_ids:
        target_terms = term_vectors[target_id]
        for key in analyzed_q:
            if key not in target_terms:
                continue
            else:
                key_term_vec = target_terms[key]
                ttf = key_term_vec['ttf']
                tf = key_term_vec['term_freq']
                df = key_term_vec['doc_freq']
                dl = target_terms['_doc']['dl']
                avg_dl = target_terms['_doc']['avg_dl']

                OTF = compute_otf(tf, dl, avg_dl)
                TFIDF = compute_tfidf(tf, dl, avg_dl, D, df)
                BM = compute_bm(tf, dl, avg_dl, D, df)
                LML = compute_lml(tf, dl, V)
                LMJM = compute_lmjm(tf, dl, V, ttf, _lambda)

                doc_score_TFIDF = add_score(target_id, doc_score_TFIDF, TFIDF)
    # top 1000 docs
    query_doc_TFIDF = get_top_1000(doc_score_TFIDF)
    # write results into files
    for i in range(1, 1001):
        if i <= len(query_doc_TFIDF):
            write_score_file("TFIDF_" + version + ".txt", query_doc_TFIDF, query_id)

# PSEUDO RELEVANCE ELASTICSEARCH AGGREGATION
#############################################################################################


queries = []
queries_id = []
with open("./queries_short.txt", "r") as f:
    for i in f.readlines():
        queries.append(re.findall("[A-Z|a-z].*[A-Z|a-z]", i))
        queries_id.append(re.findall("^[0-9]+", i)[0])

query_sig = {}
for idx, q in enumerate(queries):
    query_id = queries_id[idx]
    unanalyzed_q = re.split(" |-", q[0])
    analyzed_q = analyze_word(q[0])
    # different model score
    query_sig_list = []
    query_sig_dict = {}
    for key in analyzed_q:
        significant = es.search(index="ap_files_stop_stem",
                                body={
                                    "size": 0,
                                    "query": {
                                        "match": {"content": key}
                                    },
                                    "aggregations": {
                                        "significant_words": {
                                            "significant_terms": {"field": "content"}
                                        }
                                    }
                                })['aggregations']['significant_words']['buckets']
        for word in significant:
            query_sig_dict[word['key']] = word['score']
    query_sig_list += sorted(query_sig_dict, key=query_sig_dict.get, reverse=True)
    query_sig[query_id] = query_sig_list

with open("./interesting_es.json", "w") as f:
    json.dump(query_sig, f)
