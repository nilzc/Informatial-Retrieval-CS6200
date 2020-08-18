from crawler import Crawler
from es import ES

# given seed URLs, our topic is "Catholic Church"
seed_urls = ["http://en.wikipedia.org/wiki/Catholic_Church",
             "http://en.wikipedia.org/wiki/Christianity",
             "http://en.wikipedia.org/wiki/Ten_Commandments_in_Catholic_theology"
             ]

# crawler
crawler = Crawler()
crawler.initialize(seed_urls)
crawler.crawl_control()

# merge indexes
my_es = ES()
my_es.initialize()
my_es.es_control()
