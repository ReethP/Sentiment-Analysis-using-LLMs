# Keyword extraction tools
from rake_nltk import Rake
import yake
# from summa import keywords #For TextRank
from keybert import KeyBERT

# Text summarization tools

# Why use text summarization tools? According to multiple studies, average user reviews should be around 200 characters on average. If it exceeds that
# The text needs to be summarized and keywords need to be extracted. I think there's a study out there that longer user reviews has a correlation with negative sentiment so I'm going to tag all reviews longer than 300 characters and look at data

# Edit: found a study https://www.researchgate.net/publication/324507540_Reviews'_length_and_sentiment_as_correlates_of_online_reviews'_ratings
# "The results also show that consumer review sentiment correlates positively and review length correlates negatively with consumer online review ratings"

text = "ADDRESSED MY CONCERN PROPERLY, VERY HELPFUL AND PATIENT. KUDOS TO YOUR BEST EMPLOYEE"
# ----RAKE
# Uses stopwords for english from NLTK, and all puntuation characters by
# default
r = Rake()

# Extraction given the text.
print("Rake: ")
r.extract_keywords_from_text(text)

# Extraction given the list of strings where each string is a sentence.
# r.extract_keywords_from_sentences()

# To get keyword phrases ranked highest to lowest.
r.get_ranked_phrases()

# To get keyword phrases ranked highest to lowest with scores.
print(r.get_ranked_phrases_with_scores())



# ----YAKE
kw_extractor = yake.KeywordExtractor()
keywords = kw_extractor.extract_keywords(text)

print("YAKE: ")
for kw in keywords:
	print(kw)

print("\n\n")

# YAKE WITH MODIFIED PARAMETERS
# language = "en"
# max_ngram_size = 3
# deduplication_threshold = 0.9
# deduplication_algo = 'seqm'
# windowSize = 1
# numOfKeywords = 20

# custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
# keywords = custom_kw_extractor.extract_keywords(text)

# for kw in keywords:
#     print(kw)


# ---- TextRank
from summa import keywords

TR_keywords = keywords.keywords(text, scores=True, words=3)
print("TextRank")
print(TR_keywords)



# keybert
kw_model = KeyBERT(model='all-mpnet-base-v2')

keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', highlight=False, top_n=10)

keywords_list= list(dict(keywords).keys())

print("\nKeyBert: ",keywords_list)