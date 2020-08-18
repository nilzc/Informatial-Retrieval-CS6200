from elasticsearch import Elasticsearch
from elasticsearch import helpers
import numpy as np
import math
import os
import json


class ES:

    def __init__(self):
        # this is the info to interact with Elasticsearch cloud server, already expired
        self.hosts = ["https://e677ecbec38f4ca6a93a6538d6dc2918.northamerica-northeast1.gcp.elastic-cloud.com:9243/"]
        self.cloud_id = "ZC_CS6200:bm9ydGhhbWVyaWNhLW5vcnRoZWFzdDEuZ2NwLmVsYXN0aWMtY2xvdWQuY29tJGU2NzdlY2JlYzM4ZjRjYTZhOTNhNjUzOGQ2ZGMyOTE4JGUyYmZiMjAwNjY2NjRlOTVhNjc0ZWY0OWE5ODBhMzkz"
        self.index = "hw3"
        self.es = Elasticsearch(hosts=self.hosts, timeout=60, clould_id=self.cloud_id, http_auth=('elastic', 'rlj3NbyVqLOIUKKHhH4OGAjC'))
        self.in_links = {}
        self.out_links = {}

    def get_in_links(self):
        all_docs = helpers.scan(self.es,
                                index=self.index,
                                query={
                                    "query": {
                                        "match_all": {}
                                    },
                                    "_source": ["in_links"]
                                },
                                size=2000,
                                request_timeout=30)
        count = 0
        for i in all_docs:
            count += 1
            print(count)
            url = i["_id"]
            in_links = i["_source"]["in_links"]
            self.in_links[url] = in_links

    def write_in_links(self):
        with open("./links/in_links.txt", "a") as f:
            for url in self.in_links:
                line = "{} ".format(url)
                for l in self.in_links[url]:
                    line += "{} ".format(l)
                f.write(line)
                f.write("\n")

    def get_out_links(self):
        all_docs = helpers.scan(self.es,
                                index=self.index,
                                query={
                                    "query": {
                                        "match_all": {}
                                    },
                                    "_source": ["out_links"]
                                },
                                size=2000,
                                request_timeout=30)
        count = 0
        for i in all_docs:
            count += 1
            print(count)
            url = i["_id"]
            out_links = i["_source"]["out_links"]
            self.out_links[url] = out_links

    def write_out_links(self):
        with open("./links/out_links.txt", "a", encoding="utf-8") as f:
            for url in self.out_links:
                line = "{} ".format(url)
                for l in self.out_links[url]:
                    line += "{} ".format(l)
                f.write(line)
                f.write("\n")


my_es = ES()
my_es.get_out_links()


class PageRank():

    def __init__(self):
        self.M = {}
        self.P = []
        self.N = 0
        self.out_links = {}

        self.S = []
        self.L = {}
        self.d = 0.85
        self.PR = {}

        self.initialize()

    def initialize(self):
        self.read_in_links()
        self.read_out_links()
        self.P = [i for i in self.M]
        self.N = len(self.P)
        self.get_S()
        self.get_L()

    def get_page_rank(self):
        for i in self.P:
            self.PR[i] = 1/self.N

        newPR = {}
        perplexity = 0
        loops = 0
        unit_no_change = 0
        while True:
            loops += 1
            print(loops)

            sinkPR = 0
            for p in self.S:
                sinkPR += self.PR[p]
            for p in self.P:
                # 80% * sum(I(Ai) / out_degree(Ai)) + 20% * 1 / the # of pages
                newPR[p] = (1 - self.d) / self.N
                newPR[p] += self.d * sinkPR / self.N
                for q in self.M[p]:
                    # sum(I(Ai) / out_degree(Ai)), all the importance
                    newPR[p] += self.d * self.PR[q] / self.L[q]
            for p in self.P:
                self.PR[p] = newPR[p]
            new_perplexity = 2 ** (-np.sum([self.PR[x] * math.log(self.PR[x], 2) for x in self.PR]))
            if int(perplexity) % 10 == int(new_perplexity) % 10:
                unit_no_change += 1
            else:
                unit_no_change = 0
            perplexity = new_perplexity
            print(unit_no_change, perplexity)
            if unit_no_change == 4:
                return

    def get_S(self):
        for p in self.P:
            if p not in self.out_links:
                self.S.append(p)

    def get_L(self):
        for p in self.P:
            if p in self.out_links:
                self.L[p] = len(self.out_links[p])
            else:
                self.L[p] = 0

    def print_top_500(self):
        if os.path.exists("./links/page_rank.txt"):
            os.remove("./links/page_rank.txt")
        final = sorted(self.PR, key=self.PR.get, reverse=True)[:500]
        adjust = 160
        lines = [[p.ljust(adjust), str(self.PR[p]).ljust(adjust), str(self.L[p]).ljust(adjust),
                  str(len(self.M[p])).ljust(adjust)] for p in final]
        headers = ['Page'.ljust(adjust), 'Page Rank'.ljust(adjust), 'No. of Outlinks'.ljust(adjust),
                   'No. of Inlinks'.ljust(adjust)]
        with open('./links/page_rank.txt', "a") as f:
            f.write(''.join(headers))
            f.write("\n")
            for l in lines:
                f.write(''.join(l))
                f.write('\n')

    def read_in_links(self):
        # with open('./links/new_in_links.json', "r") as f:
        #     for line in f.readlines():
        #         line = json.loads(line)
        #         for i in line:
        #             self.M[i] = line[i]
        with open("./links/new_in_links.txt", "r") as f:
            for line in f.readlines():
                new_line = line.replace(" \n", "")
                new_line = new_line.replace("\n", "")
                new_line = new_line.split(" ")
                if len(new_line) == 1:
                    self.M[new_line[0]] = []
                else:
                    self.M[new_line[0]] = new_line[1:]

    def read_out_links(self):
        with open("./links/out_links.txt", "r", encoding="utf-8") as f:
            for line in f.readlines():
                new_line = line.replace(" \n", "")
                new_line = new_line.replace("\n", "")
                new_line = new_line.split(" ")
                if len(new_line) == 1:
                    self.out_links[new_line[0]] = []
                else:
                    self.out_links[new_line[0]] = new_line[1:]


my_pr = PageRank()
my_pr.get_page_rank()
my_pr.print_top_500()
