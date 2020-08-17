# Objectives
1. Implement and compare various retrieval systems using vector space models and language models
2. Getting familiar with Elasticsearch: one of the available commercial-grade indexes
3. This assignment will include:
    * A program to parse the corpus and index it with Elasticsearch
    * A query processor, which runs queries from an input file using a selected retrieval model

# Indexing and Model Building
1. Read all documents line by line, and extract only text content by identifying the document pattern.
2. Create a new index in Elasticseach with a customized template (stop words and stemmer).
3. Let Elasticsearch do the indexing, and use ```term_vectors``` API to retrieve all the statistical info (term frequency, doc frequency, etc.)
4. Build retrieval models based on the statistical info above.
5. Run the given queries, and rank all docs based on their relevance score (from each model)
6. Run ```trec_eval.pl``` to evaluate model performance. 

# Pseudo-relevance Feedback
1. Retrieve top k documents using the models above.
2. Identify interesting terms (heplful to retrieve relevant docs) in the documents.
3. Add the terms to the queries, and re-run the retrieval program.

* Note: the identifying process must be done by a program, not manually.*
