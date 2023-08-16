import csv
import time
from collections import namedtuple
from keyword_extraction import extract_keywords
from topic_identification import perform_lda, perform_nmf
from functools import partial
from pos_tagging import initialize_pos_models, tag_row


Pos_Header = namedtuple('Row', [
	'ID', 'Review', 'Annotated_Sentiment',
	'Nouns_MRM_Bert', 'Verbs_MRM_Bert', 'Adjectives_MRM_Bert', 'Others_MRM_Bert',
	'Nouns_Vblagoje_Bert', 'Verbs_Vblagoje_Bert', 'Adjectives_Vblagoje_Bert', 'Others_Vblagoje_Bert',
	'Nouns_QCRI_Bert', 'Verbs_QCRI_Bert', 'Adjectives_QCRI_Bert', 'Others_QCRI_Bert',
	'Nouns_TweebankNLP_Bert', 'Verbs_TweebankNLP_Bert', 'Adjectives_TweebankNLP_Bert', 'Others_TweebankNLP_Bert'
])

Zero_Shot_Header = namedtuple('Row', [
	'ID', 'Review', 'Annotated_Sentiment',
	'Labels_MoritzMNLI_Deberta',
	'Labels_MoritzMulti_Deberta',
	'Labels_Crossencoder_Roberta',
	'Labels_Valhalla_Distilbart',
	'Labels_Narsil_Deberta',
	'Labels_Crossencoder_Deberta'
])

def main():
	
	models = initialize_pos_models()

	with open('input.csv', 'r') as file, open('pos.csv', 'w', newline='') as pos_output:
		
		start_time = time.time()
		reader = csv.reader(file)
		pos_writer = csv.writer(pos_output)

		# Skip header
		next(reader)
		pos_writer.writerow(list(Pos_Header._fields))

		# Populate the first three columns IDm Review, and Annotated_Sentiment with data from the file
		row_records = [Pos_Header(row[0], row[1], row[2], *[''] * (len(Pos_Header._fields) - 3)) for row in reader]

		# print(row_records)
		
		for model in models:
			tag_row_with_model = partial(tag_row, [model])
			row_records = (map(tag_row_with_model, row_records))

		pos_writer.writerows(row_records)

		end_time = time.time()
		print("Execution Time: ", end_time - start_time, " seconds")

main()