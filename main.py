import csv
import time
from collections import namedtuple
from keyword_extraction import extract_keywords
from sentiment_analysis import process_row, calculate_overall_scores, initialize_sentiment_models
from topic_identification import perform_lda, perform_nmf
from functools import partial

# Define the namedtuple - header for output files
Row = namedtuple('Row', [
    'ID', 'Review', 'Annotated_Sentiment',
    'Vader_Sentiment', 'Matches_Vader',
    'DistilBERT_Sentiment', 'Matches_DistilBERT', 'DistilBERT_Scores',
    'Finiteautomata_Sentiment', 'Matches_Finiteautomata', 'Finiteautomata_Scores',
    'Pysentimiento_Sentiment', 'Matches_Pysentimiento', 'Pysentimiento_Scores',
    'Cardiffnlp_Roberta_Sentiment', 'Matches_Cardiffnlp_Roberta', 'Cardiffnlp_Roberta_Scores',
    'Cardiffnlp_Xlm_Robert_Sentiment', 'Matches_Cardiffnlp_Xlm_Robert', 'Cardiffnlp_Xlm_Robert_Scores',
    'Seethal_Sentiment', 'Matches_Seethal', 'Seethal_Scores',
    'Bert_Nlptown_Sentiment', 'Matches_Bert_Nlptown', 'Bert_Nlptown_Rating',
    'Overall_Score', 'Overall_Sentiment', 'Matches_Annotated'
])

Keyword_Header = namedtuple('Row', ['ID','Review', 'Rake_Keywords', 'Yake_Keywords', 'TextRank_Keywords', 'KeyBERT_Keywords','Keybert_Optimized_Keywords'])

Topics_Header = namedtuple('Row', ['LDA_Topics', 'NMF_Topics'])

def main():

	keyword_records = []
	reviews = []

	models = initialize_sentiment_models()  # Initialize models using the function

	with open('input.csv', 'r') as file, open('sentiment.csv', 'w', newline='') as senitment_output, \
	open('keywords.csv', 'w', newline='') as keywords_output, open('topics.csv', 'w', newline='') as topics_output:
		start_time = time.time()
		reader = csv.reader(file)
		sentiment_writer = csv.writer(senitment_output)
		keyword_writer = csv.writer(keywords_output)
		topics_writer = csv.writer(topics_output)

		# Skip header of input file and write headers of output files
		next(reader)
		sentiment_writer.writerow(list(Row._fields))
		keyword_writer.writerow(list(Keyword_Header._fields))
		topics_writer.writerow(list(Topics_Header._fields))

		# Populate the first three columns IDm Review, and Annotated_Sentiment with data from the file
		row_records = [Row(row[0], row[1], row[2], *[''] * (len(Row._fields) - 3)) for row in reader]

		# Process data. partial() makes it easier to use the many different models to use the same function
		for model in models:
		    process_row_with_model = partial(process_row, [model])
		    row_records = (map(process_row_with_model, row_records))

		# Apple the function calculate_overall_scores to all values in row_records
		row_records = map(calculate_overall_scores, row_records)


		# Perform keyword extraction and put it into a variable while extrating the review from each row.
		# This is to be used in topic identification using LDA and NMF. Every line in the row_records
		# is also written into the excel file
		for row in row_records:
			keyword_records.append(extract_keywords(row.ID,row.Review))
			reviews.append(row.Review)
			sentiment_writer.writerow(row)

		lda_topics = perform_lda(reviews)
		nmf_topics = perform_nmf(reviews)

		# write the topics identified by LDA and NMF into its own file
		for iterate in range(len(lda_topics)):
			topics_writer.writerow([lda_topics[iterate],nmf_topics[iterate]])

		# Write the identified keywords and pos into an excel file
		keyword_writer.writerows(keyword_records)

		end_time = time.time()
		print("Execution Time: ", end_time - start_time, " seconds")

main()