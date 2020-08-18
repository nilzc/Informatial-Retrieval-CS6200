import os
import re
from nltk.stem.snowball import SnowballStemmer
import json
import math
import gzip
import io


# PREPARATION
#############################################################################################
# file parser
def file_parser():
    file_name = os.listdir("./ap89_collection/")
    file_name = ["./ap89_collection/" + i for i in file_name]

    # needed objects
    files = {}
    file = list()
    add_file_flag = 0

    txt = list()
    txt_item = list()
    txt_flag = 0

    # read each file
    for f in file_name:
        f_open = open(f)
        data = f_open.readlines()
        f_open.close()

        for line in data:
            line = line.strip()
            # file end
            if re.search("</DOC>", line):
                add_file_flag = 0
                files[data_id] = ' '.join(file)
                file = list()
            # add lines to file
            if add_file_flag == 1:
                # id
                if re.search("</DOCNO>", line):
                    data_id = re.sub("(<DOCNO> )|( </DOCNO>)", "", line)
                # text
                # text end
                if re.search("</TEXT>", line):
                    txt_flag = 0
                if txt_flag == 1:
                    file.append(line)
                # text start
                if re.search("<TEXT>", line):
                    if re.search("[A-Z|a-z]*[a-z]", line):
                        file.append(line[6:])
                    txt_flag = 1
            # file start
            if re.search("<DOC>", line):
                add_file_flag = 1
    return files


files = file_parser()
doc_ids = [d for d in files]

instances = [{
    "id": d,
    "content": files[d]
} for d in files]
files = []

# read stop words
with open("./stoplist.txt", "r") as f:
    stop_words = f.read().split("\n")

# MAIN
#############################################################################################


class MyIndex:

    def __init__(self, name, stop_words, is_stem, is_compress):

        self.name = name
        self.is_stem = is_stem
        self.is_compress = is_compress
        self.folder_name = self.folder_name()
        self.terms_map = {}
        self.terms_map_rev = {}
        self.terms_id = 0
        self.terms_df = {}
        self.docs_map = {}
        self.docs_map_rev = {}
        self.docs_id = 0
        self.docs_dl = {}
        self.terms_ttf = {}
        self.terms_tf = {}
        self.field_info = {
            'D': 0,
            'V': 0,
            'total_ttf': 0
        }
        self.doc_count = 0
        self.partial = 0
        self.catalog = {}
        self.new_catalog = {}
        self.tracker = 0
        self.analyzer = Analyzer(stop_words, is_stem)
        print("Index Created")

    def folder_name(self):
        if self.is_stem and self.is_compress:
            return "with_stem_compress"
        elif self.is_stem and not self.is_compress:
            return "with_stem_no_compress"
        elif not self.is_stem and self.is_compress:
            return "no_stem_compress"
        else:
            return "no_stem_no_compress"

#############################################################################################
    def insert_doc(self, instance):
        doc = Doc(instance, self.analyzer)
        self.doc_count += 1
        if len(doc.tokens) == 0:
            return
        self.update_terms_map(doc)
        self.update_docs_info(doc)
        self.update_df(doc)
        self.update_ttf(doc)
        self.update_info(doc)
        self.get_tf_pos(doc)
        self.start_over()
        self.finish()
        print(self.field_info['D'])

    def update_terms_map(self, doc):
        for term in [t[0] for t in doc.tokens]:
            if term not in self.terms_map:
                self.terms_map[term] = self.terms_id
                self.terms_id += 1
    
    def update_docs_info(self, doc):
        if doc.id not in self.docs_map:
            self.docs_map[doc.id] = self.docs_id
            self.docs_dl[str(self.docs_id)] = doc.dl
            self.docs_id += 1

    def update_info(self, doc):
        self.field_info['D'] += 1
        self.field_info['V'] = len(self.terms_df)
        self.field_info['total_ttf'] += len(doc.tokens)

    def update_df(self, doc):
        for token in set([t[0] for t in doc.tokens]):
            if token in self.terms_df:
                self.terms_df[token] += 1
            else:
                self.terms_df[token] = 1

    def update_ttf(self, doc):
        for token in [t[0] for t in doc.tokens]:
            if token in self.terms_ttf:
                self.terms_ttf[token] += 1
            else:
                self.terms_ttf[token] = 1

    def get_tf_pos(self, doc):
        for token in doc.tokens:
            term_id = self.terms_map[token[0]]
            doc_id = self.docs_map[token[1]]
            pos = token[2]
            if term_id in self.terms_tf:
                if doc_id in self.terms_tf[term_id]:
                    self.terms_tf[term_id][doc_id].append(pos)
                else:
                    self.terms_tf[term_id][doc_id] = [pos]
            else:
                self.terms_tf[term_id] = {}
                self.terms_tf[term_id][doc_id] = [pos]

    # convert dict to "," separated format
    def output_format(self, dt):
        doc_ids = [key for key in dt]
        final_line = ''
        for doc_id in doc_ids:
            pos = dt[doc_id]
            line = str(doc_id)
            for p in pos:
                line = line + ',' + str(p)
            line = line + ' '
            final_line = final_line + line
        return final_line

    def gzip_compress(self, string):
        return gzip.compress(string.encode())

    def gzip_decompress(self, bytes_obj):
        return gzip.decompress(bytes_obj).decode()

    # add 1000 docs limit
    def start_over(self):
        if self.doc_count == 1000:
            catalog = {}
            start = 0
            if self.is_compress:
                # with gzip.open("./" + self.name + "/" + self.folder_name + "/merge/partial_" + str(self.partial),
                #           "ab") as f:
                with open("./" + self.name + "/" + self.folder_name + "/merge/partial_" + str(self.partial),
                               "ab") as f:
                    for key in self.terms_tf:
                        inv_list = self.terms_tf[key]
                        inv_list = self.sort_doc(inv_list)
                        dump_line = self.output_format(inv_list)
                        f.write(self.gzip_compress(dump_line))
                        # f.write(dump_line.encode())
                        length = f.tell() - start
                        catalog[str(key)] = [start, length]
                        start += length
            else:
                with open("./" + self.name + "/" + self.folder_name + "/merge/partial_" + str(self.partial), "a") as f:
                    for key in self.terms_tf:
                        inv_list = self.terms_tf[key]
                        inv_list = self.sort_doc(inv_list)
                        dump_line = self.output_format(inv_list)
                        f.write(dump_line)
                        length = f.tell() - start
                        catalog[str(key)] = [start, length]
                        start += length
            self.catalog[str(self.partial)] = catalog
            self.terms_tf = {}
            self.partial += 1
            self.doc_count = 0

    # sort doc blocks by term frequency
    def sort_doc(self, dt):
        sorted_keys = sorted(dt, key=lambda k: len(dt[k]))
        result = {k: dt[k] for k in sorted_keys}
        return result

    # when all inserts are finished
    def finish(self):
        if self.field_info['D'] == 84660:
            catalog = {}
            start = 0
            if self.is_compress:
                # with gzip.open("./" + self.name + "/" + self.folder_name + "/merge/partial_" + str(self.partial),
                #           "ab") as f:
                with open("./" + self.name + "/" + self.folder_name + "/merge/partial_" + str(self.partial),
                               "ab") as f:
                    for key in self.terms_tf:
                        inv_list = self.terms_tf[key]
                        sorted_doc_ids = self.sort_doc(inv_list)
                        inv_list = {id: inv_list[id] for id in sorted_doc_ids}
                        dump_line = self.output_format(inv_list)
                        f.write(self.gzip_compress(dump_line))
                        # f.write(dump_line.encode())
                        length = f.tell() - start
                        catalog[str(key)] = [start, length]
                        start += length
            else:
                with open("./" + self.name + "/" + self.folder_name + "/merge/partial_" + str(self.partial), "a") as f:
                    for key in self.terms_tf:
                        inv_list = self.terms_tf[key]
                        sorted_doc_ids = self.sort_doc(inv_list)
                        inv_list = {id: inv_list[id] for id in sorted_doc_ids}
                        dump_line = self.output_format(inv_list)
                        f.write(dump_line)
                        length = f.tell() - start
                        catalog[str(key)] = [start, length]
                        start += length
            self.catalog[str(self.partial)] = catalog
            self.terms_tf = {}
            self.partial += 1

            self.docs_map_rev = {self.docs_map[k]: k for k in self.docs_map}
            with open("./" + self.name + "/" + self.folder_name + "/term_df.json", "w") as f:
                json.dump(self.terms_df, f)
            with open("./" + self.name + "/" + self.folder_name + "/term_ttf.json", "w") as f:
                json.dump(self.terms_ttf, f)
            with open("./" + self.name + "/" + self.folder_name + "/field_info.json", "w") as f:
                json.dump(self.field_info, f)
            with open("./" + self.name + "/" + self.folder_name + "/catalog.json", "w") as f:
                json.dump(self.catalog, f)
            with open("./" + self.name + "/" + self.folder_name + "/terms_map.json", "w") as f:
                json.dump(self.terms_map, f)
            with open("./" + self.name + "/" + self.folder_name + "/docs_map.json", "w") as f:
                json.dump(self.docs_map, f)
            with open("./" + self.name + "/" + self.folder_name + "/docs_dl.json", "w") as f:
                json.dump(self.docs_dl, f)

#############################################################################################
    def merge_control(self):
        left = '0'
        right = '1'
        if self.is_compress:
            self.new_catalog = self.merge_compress(left, right, self.catalog[left], self.catalog[right])
        else:
            self.new_catalog = self.merge(left, right, self.catalog[left], self.catalog[right])
        for n in range(2, 85):
            left = str(int(left) + 100)
            right = str(n)
            if n > 2:
                os.remove("./" + self.name + "/" + self.folder_name + "/merge/partial_" + str(int(left) - 100))
            if self.is_compress:
                new_catalog = self.merge_compress(left, right, self.new_catalog, self.catalog[right])
            else:
                new_catalog = self.merge(left, right, self.new_catalog, self.catalog[right])
            self.new_catalog = new_catalog
        os.remove("./" + self.name + "/" + self.folder_name + "/merge/partial_" + left)
        with open("./" + self.name + "/" + self.folder_name + "/merge/new_catalog.json", "w") as f:
            json.dump(self.new_catalog, f)

    # merge 2 inverted lists
    def merge(self, left, right, left_catalog, right_catalog):
        print(left)
        left = str(left)
        right = str(right)
        new_start = 0
        new_catalog = {}
        for term_id in left_catalog:
            left_offset = left_catalog[term_id][0]
            left_length = left_catalog[term_id][1]
            if term_id not in right_catalog:
                with open("./" + self.name + "/" + self.folder_name + "/merge/partial_" + left, "r") as f_left:
                    f_left.seek(left_offset)
                    new_line = f_left.read(left_length)
                with open("./" + self.name + "/" + self.folder_name + "/merge/partial_" + str(int(left) + 100), "a") as f_new:
                    f_new.write(new_line)
                    new_length = f_new.tell()
                new_catalog[term_id] = [new_start, new_length]
                new_start += new_length
            if term_id in right_catalog:
                right_offset = right_catalog[term_id][0]
                right_length = right_catalog[term_id][1]
                with open("./" + self.name + "/" + self.folder_name + "/merge/partial_" + left, "r") as f_left:
                    f_left.seek(left_offset)
                    left_line = f_left.read(left_length)
                with open("./" + self.name + "/" + self.folder_name + "/merge/partial_" + right, "r") as f_right:
                    f_right.seek(right_offset)
                    right_line = f_right.read(right_length)
                update_line = self.update_partial_term(left_line, right_line)
                new_line = self.output_format(update_line)
                with open("./" + self.name + "/" + self.folder_name + "/merge/partial_" + str(int(left) + 100), "a") as f_new:
                    f_new.write(new_line)
                    new_length = f_new.tell()
                new_catalog[term_id] = [new_start, new_length]
                new_start += new_length
                del right_catalog[term_id]
        for term_id in right_catalog:
            right_offset = right_catalog[term_id][0]
            right_length = right_catalog[term_id][1]
            with open("./" + self.name + "/" + self.folder_name + "/merge/partial_" + right, "r") as f_right:
                f_right.seek(right_offset)
                new_line = f_right.read(right_length)
            with open("./" + self.name + "/" + self.folder_name + "/merge/partial_" + str(int(left) + 100), "a") as f_new:
                f_new.write(new_line)
                new_length = f_new.tell()
            new_catalog[term_id] = [new_start, new_length]
            new_start += new_length
        return new_catalog

    # for compress version
    def merge_compress(self, left, right, left_catalog, right_catalog):
        print(left)
        left = str(left)
        right = str(right)
        new_catalog = {}
        new_start = 0
        for term_id in left_catalog:
            left_offset = left_catalog[term_id][0]
            left_length = left_catalog[term_id][1]
            if term_id not in right_catalog:
                with open("./" + self.name + "/" + self.folder_name + "/merge/partial_" + left, "rb") as f_left:
                    f_left.seek(left_offset)
                    new_line = f_left.read(left_length)
                with open("./" + self.name + "/" + self.folder_name + "/merge/partial_" + str(int(left) + 100), "ab") as f_new:
                    new_length = f_new.write(new_line)
                new_catalog[term_id] = [new_start, new_length]
                new_start += new_length
            if term_id in right_catalog:
                right_offset = right_catalog[term_id][0]
                right_length = right_catalog[term_id][1]
                with open("./" + self.name + "/" + self.folder_name + "/merge/partial_" + left, "rb") as f_left:
                    f_left.seek(left_offset)
                    left_line = f_left.read(left_length)
                    left_line = self.gzip_decompress(left_line)
                with open("./" + self.name + "/" + self.folder_name + "/merge/partial_" + right, "rb") as f_right:
                    f_right.seek(right_offset)
                    right_line = f_right.read(right_length)
                    right_line = self.gzip_decompress(right_line)
                update_line = self.update_partial_term(left_line, right_line)
                new_line = self.output_format(update_line)
                with open("./" + self.name + "/" + self.folder_name + "/merge/partial_" + str(int(left) + 100), "ab") as f_new:
                    new_length = f_new.write(self.gzip_compress(new_line))
                new_catalog[term_id] = [new_start, new_length]
                new_start += new_length
                del right_catalog[term_id]
        for term_id in right_catalog:
            right_offset = right_catalog[term_id][0]
            right_length = right_catalog[term_id][1]
            with open("./" + self.name + "/" + self.folder_name + "/merge/partial_" + right, "rb") as f_right:
                f_right.seek(right_offset)
                new_line = f_right.read(right_length)
            with open("./" + self.name + "/" + self.folder_name + "/merge/partial_" + str(int(left) + 100), "ab") as f_new:
                new_length = f_new.write(new_line)
            new_catalog[term_id] = [new_start, new_length]
            new_start += new_length
        return new_catalog

    # update inverted list for shared term, similar to merge_sort
    def update_partial_term(self, left, right):
        left = self.convert_to_dict(left)
        right = self.convert_to_dict(right)
        # left.update(right)
        left_keys = [k for k in left]
        right_keys = [k for k in right]
        i = j = 0
        result = {}
        while i < len(left_keys) and j < len(right_keys):
            if len(left[left_keys[i]]) < len(right[right_keys[j]]):
                result[left_keys[i]] = left[left_keys[i]]
                i += 1
            else:
                result[right_keys[j]] = right[right_keys[j]]
                j += 1
        if i == len(left_keys):
            result.update({k: right[k] for k in right_keys[j:]})
        elif j == len(right_keys):
            result.update({k: left[k] for k in left_keys[i:]})
        return result
        # return left

    # convert output "," separated format to dict
    def convert_to_dict(self, output_format):
        record = output_format.split(' ')
        result = {}
        for doc in record:
            if len(doc) == 0:
                continue
            doc = doc.split(',')
            doc_id = int(doc[0])
            pos = [int(i) for i in doc[1:]]
            result[doc_id] = pos
        return result

#############################################################################################
    def search(self, query, model, is_proximity):
        q = Query(query, self.analyzer)
        tokens = sorted(q.tokens, key=lambda t: self.terms_df[t])
        tokens_id = [self.terms_map[t] for t in tokens]
        doc_score = {}
        # for VSM
        if model in ["BM", "TFIDF"]:
            term_docs = {}
            for idx, t in enumerate(tokens_id):
                postings = self.read_postings(t)
                term_docs[t] = postings
                for doc_id in postings:
                    score = self.compute_score(tokens[idx], postings, doc_id, model)
                    if doc_id in doc_score:
                        doc_score[doc_id] += score
                    else:
                        doc_score[doc_id] = score
            if is_proximity:
                doc_score = self.proximity_search(doc_score, term_docs)
            doc_score = self.get_top_1000(doc_score)
            for i in range(1, 1001):
                if i <= len(doc_score):
                    self.write_score_file(model, doc_score, q.id, i)
        # for language model, calculate score for docs containing terms, then add score to those not containing
        elif model == "LML":
            doc_ids, token_docs = self.docs_to_run(tokens_id)
            for idx, t in enumerate(tokens_id):
                postings = self.read_postings(t)
                for doc_id in postings:
                    score = self.compute_score(tokens[idx], postings, doc_id, model)
                    if doc_id in doc_score:
                        doc_score[doc_id] += score
                    else:
                        doc_score[doc_id] = score
            for doc_id in doc_ids:
                for idx, t in enumerate(tokens_id):
                    if doc_id not in token_docs[t]:
                        score = self.compute_score(tokens[idx], {}, doc_id, model)
                        doc_score[doc_id] += score
            doc_score = self.get_top_1000(doc_score)
            for i in range(1, 1001):
                if i <= len(doc_score):
                    self.write_score_file(model, doc_score, q.id, i)

    def proximity_search(self, doc_score, term_docs):
        for doc in doc_score:
            containing_terms = []
            for t in term_docs:
                if doc in term_docs[t]:
                    containing_terms.append(t)
            if len(containing_terms) > 1:
                span_posting = []
                for t in containing_terms:
                    span_posting.append(term_docs[t][doc])
                shortest_span = self.shortest_span(span_posting)
                score = math.log(0.01 + math.exp(-shortest_span))
                doc_score[doc] += score
        return doc_score

    def shortest_span(self, span_posting):
        start = [0 for i in range(len(span_posting))]
        smallest = math.inf
        window_helper = list(range(len(span_posting)))
        # handle postings with only 1 pos
        for idx, v in enumerate(span_posting):
            if len(v) == 1:
                window_helper.remove(idx)
        while True:
            window = [s[start[idx]] for idx, s in enumerate(span_posting)]
            # find the span length
            _min = min(window)
            _max = max(window)
            span = _max - _min + 1
            if span < smallest:
                smallest = span
            # update smallest element index
            if len(window_helper) == 0:
                break
            smallest_position = [window[i] for i in window_helper]
            temp_min = min(smallest_position)
            idx = window.index(temp_min)
            start[idx] += 1
            # if it reaches the end, then update the second smallest
            if start[idx] == len(span_posting[idx]) - 1:
                window_helper.remove(idx)
        return smallest

    def write_score_file(self, file_name, doc_score, query_id, line_num):
        with open("./my_index_model_score/" + file_name + ".txt", "a") as f:
            temp_item = doc_score[line_num - 1]
            temp_id = ''.join(temp_item)
            write_line = query_id + " " + "Q0" + " " + temp_id + " " + str(line_num) + " " + str(temp_item[temp_id]) + " Exp"
            f.writelines("%s\n" % write_line)

    def get_top_1000(self, doc_score):
        doc_ids = sorted(doc_score, key=doc_score.get, reverse=True)[0:1000]
        query_doc = [{self.docs_map_rev[doc_id]: doc_score[doc_id]} for doc_id in doc_ids]
        return query_doc

    def compute_score(self, token, postings, doc_id, model):
        if doc_id in postings:
            tf = len(postings[doc_id])
        else:
            tf = 0
        ttf = self.terms_ttf[token]
        df = self.terms_df[token]
        dl = self.docs_dl[str(doc_id)]
        avg_dl = self.field_info['total_ttf'] / self.field_info['D']

        if model == "BM":
            k1 = 1.2
            b = 0.5
            BM = math.log((self.field_info['D'] + 0.5) / (df + 0.5)) \
                 * (tf + k1 * tf) / (tf + k1 * ((1 - b) + b * (dl / avg_dl)))
            return BM

        if model == "TFIDF":
            TFIDF = (tf / (tf + 0.5 + 1.5 * (dl / avg_dl))) * math.log(self.field_info['D'] / df)
            return TFIDF

        if model == "LML":
            lml = math.log((tf + 1) / (dl + self.field_info['V']))
            return lml

    def read_postings(self, token_id):
        token_id = str(token_id)
        offset = self.new_catalog[token_id][0]
        length = self.new_catalog[token_id][1]
        if self.is_compress:
            with open("./" + self.name + "/" + self.folder_name + "/merge/partial_8400", "rb") as f:
                f.seek(offset)
                line = f.read(length)
                line = self.gzip_decompress(line)
        else:
            with open("./" + self.name + "/" + self.folder_name + "/merge/partial_8400.json", "r") as f:
                f.seek(offset)
                line = f.read(length)
        return self.convert_to_dict(line)

    def docs_to_run(self, tokens_id):
        tokens_posting = []
        for t in tokens_id:
            line = self.read_postings(t)
            tokens_posting.append(line)
        doc_ids = []
        token_docs = {}
        for idx, pos in enumerate(tokens_posting):
            doc_ids += [p for p in pos]
            token_docs[tokens_id[idx]] = [p for p in pos]
        return list(set(doc_ids)), token_docs


class Doc:

    def __init__(self, file, analyzer):
        self.id = file['id']
        self.analyzer = analyzer
        self.dl = 0
        self.tokens = self.tokenize(file['content'])

    def tokenize(self, string):
        if len(string) == 0:
            return ()
        tokens = re.findall(pattern=r"\w+(?:\.?\w)*", string=string)
        stopped_tokens = []
        for to in tokens:
            # lower case
            to = to.lower()
            if self.analyzer.is_stop(to):
                continue
            else:
                stopped_tokens.append(self.analyzer.stem(to))
        self.dl = len(stopped_tokens)
        return_tuples = []
        for idx, token in enumerate(stopped_tokens):
            return_tuples.append(
                (token, self.id, idx + 1)
            )
        return return_tuples


class Query:
    
    def __init__(self, query, analyzer): 
        self.analyzer = analyzer
        self.ql = 0
        self.id = 0
        self.tokens = self.tokenize(query)
        
    def tokenize(self, string):
        if len(string) == 0:
            return []
        tokens = re.findall(pattern=r"\w+(?:\.?\w)*", string=string)
        stopped_tokens = []
        for to in tokens:
            # lower case
            to = to.lower()
            if self.analyzer.is_stop(to):
                continue
            else:
                stopped_tokens.append(self.analyzer.stem(to))
        self.ql = len(stopped_tokens)
        self.id = stopped_tokens[0]
        return stopped_tokens[1:]


class Analyzer:

    def __init__(self, stop_words, is_stem):
        self.stop_words = stop_words
        self.is_stem = is_stem
        self.stemmer = self.init_stemmer(is_stem)

    def init_stemmer(self, is_stem):
        if is_stem:
            return SnowballStemmer('english')
        else:
            return False

    def is_stop(self, token):
        if token in self.stop_words:
            return True
        else:
            return False

    def stem(self, token):
        if self.is_stem:
            return self.stemmer.stem(token)
        else:
            return token

# RUN
#############################################################################################


my_index = MyIndex("my_index", stop_words, True, False)


def read_my_index(name, file_name, is_stem, is_compress):
    my_index = MyIndex(name=name, stop_words=stop_words, is_stem=is_stem, is_compress=is_compress)
    with open("./my_index/" + file_name + "/catalog.json", "r") as f:
        my_index.catalog = json.load(f)

    with open("./my_index/" + file_name + "/term_df.json", "r") as f:
        my_index.terms_df = json.load(f)

    with open("./my_index/" + file_name + "/term_ttf.json", "r") as f:
        my_index.terms_ttf = json.load(f)

    with open("./my_index/" + file_name + "/field_info.json", "r") as f:
        my_index.field_info = json.load(f)

    with open("./my_index/" + file_name + "/terms_map.json", "r") as f:
        my_index.terms_map = json.load(f)

    with open("./my_index/" + file_name + "/docs_map.json", "r") as f:
        my_index.docs_map = json.load(f)

    with open("./my_index/" + file_name + "/docs_dl.json", "r") as f:
        my_index.docs_dl = json.load(f)

    with open("./my_index/" + file_name + "/merge/new_catalog.json", "r") as f:
        my_index.new_catalog = json.load(f)
    my_index.docs_map_rev = {my_index.docs_map[k]: k for k in my_index.docs_map}
    return my_index


def insert_docs(index, docs):
    for i in docs:
        index.insert_doc(i)


# add docs to index
insert_docs(my_index, instances)

my_index = read_my_index("my_index", "no_stem_no_compress", False, False)

# merge partial lists
my_index.merge_control()
# merge compressed version
my_index.merge_compress('0', '1', my_index.catalog['0'], my_index.catalog['1'])

# read queries
queries = []
with open("./queries.txt", "r") as f:
    for i in f.readlines():
        queries.append(i)

# search, get ranked list
for q in queries:
    my_index.search(q, "BM", False)


