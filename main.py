import csv
import time
from collections import namedtuple
from keyword_extraction import extract_keywords
from sentiment_analysis import process_row, calculate_overall_scores, initialize_models
from functools import partial

# Define the namedtuple - header for sentiment output
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
    'Overall_Score', 'Overall_Sentiment', 'Matches_Annotated',
])

def main():

	models = initialize_models()  # Initialize models using the function
	with open('input.csv', 'r') as file, open('sentiment.csv', 'w', newline='') as senitment_output, open('keywords.csv', 'w', newline='') as keywords_output:
		start_time = time.time()
		reader = csv.reader(file)
		sentiment_writer = csv.writer(senitment_output)
		keyword_writer = csv.writer(keywords_output)

		# Skip header
		next(reader)
		sentiment_writer.writerow(list(Row._fields))
		keyword_writer.writerow(['ID','Review', 'Rake Keywords', 'Yake Keywords', 'TextRank Keywords', 'KeyBERT Keywords'])

		# Populate the first three columns IDm Review, and Annotated_Sentiment with data from the file
		row_records = [Row(row[0], row[1], row[2], *[''] * (len(Row._fields) - 3)) for row in reader]

		# Process data. partial() makes it easier to use the many different models to use the same function
		for model in models:
		    process_row_with_model = partial(process_row, [model])
		    row_records = (map(process_row_with_model, row_records))

		row_records = map(calculate_overall_scores, row_records)

		sentiment_writer.writerows(row_records)  # Write all the modified rows to the output CSV

		keyword_records = []
		for row in row_records:
			keyword_records.append(extract_keywords(row.ID,row.Review))
		keyword_writer.writerows(keyword_records)

		end_time = time.time()
		print("Execution Time: ", end_time - start_time, " seconds")

main()