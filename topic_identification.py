import nltk
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_reviews(reviews):
    preprocessed_reviews = []

    for review in reviews:
        review = review.lower().replace("'", "")
        tokenized_review = word_tokenize(review)
        # preprocessed_review = [word for word in tokenized_review if word not in stop_words]
        preprocessed_reviews.append(tokenized_review)

    return preprocessed_reviews

def perform_lda(unprocessed):
    reviews = preprocess_reviews(unprocessed)
    dictionary = corpora.Dictionary(reviews)
    bow_corpus = [dictionary.doc2bow(review) for review in reviews]

    lda_model = models.LdaModel(bow_corpus, num_topics=10, id2word=dictionary, passes=10)
    topics = lda_model.print_topics(num_words=5)

    return topics

def perform_nmf(documents):
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(documents)

    num_topics = 10
    nmf_model = NMF(n_components=num_topics, random_state=42)
    document_topic_matrix = nmf_model.fit_transform(dtm)
    topic_term_matrix = nmf_model.components_

    words = vectorizer.get_feature_names_out()
    num_top_words = 5
    topics = []

    for topic_idx, topic in enumerate(topic_term_matrix):
        top_words_idx = topic.argsort()[:-num_top_words-1:-1]
        top_words = [words[i] for i in top_words_idx]
        topics.append(f"Topic {topic_idx}: {', '.join(top_words)}")

    return topics