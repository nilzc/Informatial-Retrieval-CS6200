# Objectives
1. Implement your own index to take place of Elasticsearch in HW1 (doing the same thing as Elasticsearch).
2. Limit memory or disk I/O.
3. This assignment includes 2 programs:
    * A tokenizer and indexer.
    * An updated version of HW1 ranker which uses your inverted index.

# Tokenizing
1. Produce a sequence of tokens from given documents (possibly seperated by single periods, all lowercase).
2. Assign IDs to each term and document, and maintain corresponding maps.
3. Store tuples like this ```(term_id, doc_id, position)```

# Indexing
1. Create an inverted list, which contains statistical info like term frequency, document frequency, ttf and so on.
    * Experiment with the affects of stemming and stop words removal on query performance.
2. Create partial inverted lists for all terms in a single pass through the collection (to limit disk I/O).
    * Create one file for every 1000 docs.
3. Merge partial lists to a final index.
    * In this step, maintain a file to remember each term's position in a partial list is very helpful.
    * So that you can read a file from a specific position with a certain length.
    
# Searching
1. Run the same queries from HW1 using your own index and compare the results.
2. Apply proximity search on a retireval model and observe the results.
    * Span-based proximity distance measures (min coverage)
