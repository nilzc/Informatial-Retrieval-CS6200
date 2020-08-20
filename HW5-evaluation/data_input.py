from elasticsearch import Elasticsearch


class ES:

    def __init__(self):
        self.hosts = ["https://f2ff43d409574698a747eaa43256d1e0.northamerica-northeast1.gcp.elastic-cloud.com:9243/"]
        self.cloud_id = ""
        self.index = "hw5"
        self.es = Elasticsearch(hosts=self.hosts, timeout=60, clould_id=self.cloud_id, http_auth=('elastic', 'nRGUXlzD1f8kOT63iLehSG9a'))
        self.qrel = {"151901": {}, "151902": {}, "151903": {}}
        self.qrel_raw = {"151901": {}, "151902": {}, "151903": {}}
        self.qrel_temp = {}
        self.rank_list = {"151901": [], "151902": [], "151903": []}
        self.query = ["College of Cardinals", "Ten Commandments", "Recent Popes"]
        self.query_id = ["151901", "151902", "151903"]

    def get_qrel(self):
        temp = self.es.search(index=self.index,
                                        body={
                                            "query": {
                                               "match_all": {}
                                            }
                                        })['hits']['hits']
        for item in temp:
            key = item['_id']
            value = item['_source']['relevance']
            self.qrel_temp[key] = value

    def get_rank_list(self):
        for idx, q in enumerate(self.query):
            print("Reading ranked list: " + str(idx+1))
            q_id = self.query_id[idx]
            temp = self.es.search(index="hw3",
                                  body={
                                      "size": 1200,
                                      "query": {
                                        "match": {
                                            "text_content": q
                                        }
                                      },
                                      "_source": ""
                                  })['hits']['hits']
            for item in temp:
                self.rank_list[q_id].append({item['_id']: item['_score']})
            # i = 0
            # for item in temp:
            #     if i <= 200:
            #         if item['_id'] in self.qrel[q_id]:
            #             self.rank_list[q_id].append({item['_id']: item['_score']})
            #         i += 1
            #     else:
            #         self.rank_list[q_id].append({item['_id']: item['_score']})

    def output_qrel(self):
        for q_id in self.query_id:
            record = "Yiyun_Zhu, " + q_id
            for key in self.qrel_temp[record]:
                s1 = self.qrel_temp[record][key]
                s2 = self.qrel_temp["Zhuocheng_Lin,+" + q_id][key]
                s3 = self.qrel_temp["Jiayi_Liu, " + q_id][key]
                final_s = (s1 + s2 + s3) / 3
                self.qrel_raw[q_id][key] = final_s
                if final_s < 1:
                    self.qrel[q_id][key] = 0
                else:
                    self.qrel[q_id][key] = 1

        with open("./data/qrel.txt", "a", encoding="utf-8") as f:
            for q_id in self.qrel:
                for doc in self.qrel[q_id]:
                    rel = self.qrel[q_id][doc]
                    line = "{0} 0 {1} {2}\n".format(q_id, doc, rel)
                    f.write(line)

        with open("./data/qrel_raw.txt", "a", encoding="utf-8") as f:
            for q_id in self.qrel_raw:
                for doc in self.qrel_raw[q_id]:
                    rel = self.qrel_raw[q_id][doc]
                    line = "{0} 0 {1} {2}\n".format(q_id, doc, rel)
                    f.write(line)

    def output_rank_list(self):
        with open("./data/ranked_list.txt", "a", encoding="utf-8") as f:
            for q_id in self.rank_list:
                for idx, item in enumerate(self.rank_list[q_id]):
                    for url in item:
                        line = '{0} Q0 {1} {2} {3} Exp\n'.format(q_id, url, idx+1, str(item[url]))
                        f.write(line)
            

my_es = ES()
my_es.get_qrel()
my_es.output_qrel()
my_es.get_rank_list()
my_es.output_rank_list()

