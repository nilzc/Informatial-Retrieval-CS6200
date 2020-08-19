# Objectives
1. Given queries for your topical crawl, manully assess the relevance of retrieved documents by using your vertical search engine.
2. Code up the IR evaluation measures (essentially rewriting ```trec-eval.pl```).

# Document Assessment
1. Create a web interface to display the topic/query and the retrieved documents.
2. You should be able to give each docuemnt a 3-scale grade "non-relevant", "relevant" and "very relevant" (0, 1, 2).
    * Use an input checkboxes, radio boxes, dropdown list and so on.
3. Each student has to manually assess about 200 documents for each query.

# Write your own trec_eval
1. Input: a ranked list file and QREL file, both in TREC format.
2. Run your trec_eval on the files you're using in HW1 and make sure the results are correct.

# Precision-Recall Curves
For each one of the queries, create a precision-recall plot.
