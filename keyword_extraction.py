import csv
import time
from collections import namedtuple
from functools import partial
from rake_nltk import Rake
import yake
from keybert import KeyBERT
from summa import keywords

# Define the named tuple
Row = namedtuple('Row', ['raw_review', 'rake_keywords', 'yake_keywords', 'textrank_keywords', 'keybert_keywords'])

# Keyword extraction tools
r = Rake()
kw_extractor = yake.KeywordExtractor()
kw_model = KeyBERT(model='all-mpnet-base-v2')

# For later development:
# Text summarization models / tools for reviews with more than 300 characters. 
# Two columns to be added, specifically: More_than_300 ; bool, summary ; string

# Why use text summarization tools? According to multiple studies, average user reviews should be around 200 characters on average. If it exceeds that
# The text needs to be summarized and keywords need to be extracted. I think there's a study out there that longer user reviews has a correlation with negative sentiment so I'm going to tag all reviews longer than 300 characters and look at data

# Edit: found a study https://www.researchgate.net/publication/324507540_Reviews'_length_and_sentiment_as_correlates_of_online_reviews'_ratings
# "The results also show that consumer review sentiment correlates positively and review length correlates negatively with consumer online review ratings"

with open('input.csv', 'r') as file, open('output_keywords.csv', 'w', newline='') as outfile:
    start_time = time.time()
    reader = csv.reader(file)
    writer = csv.writer(outfile)

    # Writing the header of the CSV
    writer.writerow(list(Row._fields))

    # Populate the first two columns Review and Annotated_Sentiment with data from the file
    for row in reader:
        text = row[0]
        print(text)

        # RAKE
        r.extract_keywords_from_text(text)
        rake_keywords = ', '.join(r.get_ranked_phrases()[:3])

		# YAKE
        yake_keywords = ', '.join([keyword[0] for keyword in kw_extractor.extract_keywords(text)[:3]]) if len(kw_extractor.extract_keywords(text)) > 0 else ""

        # TextRank
        # Note: Sometimes TextRank fails to identify relevant keywords
        try:
            TR_keywords = ', '.join([keyword[0] for keyword in keywords.keywords(text, scores=True, words=3)]) if len(keywords.keywords(text, scores=True, words=3)) > 0 else ""
        except IndexError:
            TR_keywords = ""

        # KeyBERT
        keybert_keywords = ', '.join([keyword[0] for keyword in kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', highlight=False, top_n=3)])

        # Write the review and the first three keywords to the new CSV file
        writer.writerow([text, rake_keywords, yake_keywords, TR_keywords, keybert_keywords])
