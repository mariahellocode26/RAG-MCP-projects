from rank_bm25 import BM25Okapi

# sample documents
docs = ["Office equipment policy",
        "Offic furniture guidlines",
        "Office travel policy"]

# Create BM25 index
tokenized_corpus = [doc.lower().split() for doc in docs]
bm25 = BM25Okapi(tokenized_corpus)

#Define a single query and tokenize it
query = "office policy"
tokenized_query = query.lower().split() 

#Get scores for that specific query
word_scores = bm25.get_scores(tokenized_query)

#Print results
for doc, score in zip(docs, word_scores):
    print(f"Score: {score:.4f} | Doc: {doc}")