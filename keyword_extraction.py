from rake_nltk import Rake
import yake
from keybert import KeyBERT
from summa import keywords
from collections import namedtuple

# A study suggested that keyphrase vectorizers might improve the performance of BERT keywords. Performance has not be evaluated
from keyphrase_vectorizers import KeyphraseCountVectorizer

# Define the named tuple
Row = namedtuple('Row', ['ID','raw_review', 'rake_keywords', 'yake_keywords', 'textrank_keywords', 'keybert_keywords', 'keybert_optimized_keywords'])

# Keyword extraction tools and models initialization (Text rank doesn't need to be initialized)
r = Rake()
kw_extractor = yake.KeywordExtractor()
kw_model = KeyBERT(model='all-mpnet-base-v2')


def extract_keywords(id_num,text):
    r.extract_keywords_from_text(text)
    rake_keywords = ', '.join(r.get_ranked_phrases()[:3])

    yake_keywords = ', '.join([keyword[0] for keyword in kw_extractor.extract_keywords(text)[:3]]) if len(kw_extractor.extract_keywords(text)) > 0 else ""

    try:
        TR_keywords = ', '.join([keyword[0] for keyword in keywords.keywords(text, scores=True, words=3)]) if len(keywords.keywords(text, scores=True, words=3)) > 0 else ""
    except IndexError:
        TR_keywords = ""

    keybert_keywords = ', '.join([keyword[0] for keyword in kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', highlight=False, top_n=3)])

    keybert_optimized_keywords = ', '.join([keyword[0] for keyword in kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', highlight=False, top_n=3, vectorizer = KeyphraseCountVectorizer())])

    return Row(id_num,text, rake_keywords, yake_keywords, TR_keywords, keybert_keywords, keybert_optimized_keywords)

