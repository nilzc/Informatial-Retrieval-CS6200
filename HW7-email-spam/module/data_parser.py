from os import listdir
from bs4 import BeautifulSoup
import re
from progressbar import ProgressBar, Bar, Percentage
import random
import json
import email


class MyParser:

    def __init__(self):
        # self.files = listdir("./trec07p/data/")
        self.spam = {}
        self.text = {}
        self.split = {}
        self.email_parser = email.parser.Parser()

    def read_html(self):
        bar = ProgressBar(widgets=["Read html: ", Bar(), Percentage()], maxval=len(self.files))
        bar.start()
        count = 0
        for p in self.files:
            with open("././trec07p/data/{}".format(p), "r", encoding="ISO-8859-1") as f:
                all_data = f.read()
            parsed_email = self.email_parser.parsestr(text=all_data)
            text = self.get_all_content(parsed_email)
            soup = BeautifulSoup(text, "lxml")
            # text = re.findall("\w+", soup.get_text())
            self.text[p] = soup.get_text()

            if random.sample([1, 1, 1, 1, 0], 1)[0]:
                self.split[p] = "train"
            else:
                self.split[p] = "test"

            count += 1
            bar.update(count)
        bar.finish()

    def read_label(self):
        with open("./trec07p/full/index", "r") as f:
            for line in f.readlines():
                line = line.replace("\n", "")
                spam, file = line.split(" ")
                file = re.findall("/\w+.\w*$", file)[0][1:]
                self.spam[file] = spam

    def read_from_local(self):
        with open("./text.json", "r") as f:
            self.text = json.load(f)
        with open("./split.json", "r") as f:
            self.split = json.load(f)
        with open("./spam.json", "r") as f:
            self.spam = json.load(f)
        print("text, spam, split: done")

    def get_all_content(self, e):
        if not e.is_multipart():
            return e.get_payload()
        else:
            text = ""
            for p in e.get_payload():
                text += self.get_all_content(p)
            return text
