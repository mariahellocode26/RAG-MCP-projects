from sklearn.feature_extraction.text import TfidfVectorizer

# sample documents
docs = ["Office equipment policy",
        "Offic furniture guidlines",
        "Office travel policy"]

# Create TF-IDF analyzer
analyzer = TfidfVectorizer()

# Find word importance scores
word_scores = analyzer.fit_transform(docs)

#Print word importance scores
print(word_scores.toarray())

#
query = "furniture"
query_scores = analyzer.transform([query])

print(f"Query scores: {query_scores.toarray()}")


