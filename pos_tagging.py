import csv
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
from collections import namedtuple, defaultdict

# I think the input for this needs to be cleaned before being fed. It seems to glitch out if there are punctuation marks near them.

# QCRI
# JJ = Adjective
# VB/VBD = Verb
# NN/NNP = Noun / Proper Noun
# some models split the tokens into half
# Kis ##met
# need to write a script that will parse this such that it will form the whole word itself

# TweebankNLP

# ADJ = adjective 
# VERB = verb
# PRON = pronoun
# NOUN = noun 
# split by @

# Same for both moritz models but split but ##

def standardize_labels(label):
    if label in ['JJ', 'ADJ']:
        return 'Adjective'
    if label in ['VB', 'VBD', 'VERB']:
        return 'Verb'
    if label in ['PRON', 'NOUN','PROPN', 'NN', 'NNP']:
        return 'Noun'
    return 'Others' # All other labels

def initialize_pos_models():

	model_configs = {
		'MoritzMNLI_Deberta': ('mrm8488/bert-spanish-cased-finetuned-pos-16-tags'),
		'MoritzMulti_Deberta': ('vblagoje/bert-english-uncased-finetuned-pos'),
		'Crossencoder_Roberta': ('QCRI/bert-base-multilingual-cased-pos-english'),
		'Valhalla_Distilbart': ('TweebankNLP/bertweet-tb2_ewt-pos-tagging')
	}

	models = []

	for model_name, config in model_configs.items():
		tokenizer = AutoTokenizer.from_pretrained(config[0])
		model = AutoModelForTokenClassification.from_pretrained(config[0])
		pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
		noun_field = f'Nouns_{model_name}'
		verb_field = f'Verbs_{model_name}'
		adjective_field = f'Adjectives_{model_name}'
		others_field = f'Others_{model_name}'
		
		models.append((pipeline, noun_field, verb_field, adjective_field, others_field))

	return models

def process_row(models, row):
    review = row.Review

    try:
        for model, noun_field, verb_field, adjective_field, others_field in models:
            # if noun_field == 'Vader_Sentiment':
            #     # Assign the label in function created from a previous implementation
            #     sentiment_label = vader_sentiment(model, review)
            #     matches_source = (sentiment_label == row.Annotated_Sentiment)
            #     row = row._replace(Vader_Sentiment=sentiment_label, Matches_Vader=matches_source)

            # elif noun_field == 'Bert_Nlptown_Sentiment':
            #     # this model is processed by another function
            #     sentiment_label, star_rating = analyze_sentiment(model(review))
            #     row = row._replace(Bert_Nlptown_Sentiment=sentiment_label, Matches_Bert_Nlptown=(sentiment_label == row.Annotated_Sentiment), Bert_Nlptown_Rating=star_rating)
            # else:
            nouns, verbs, adjectives, others = identify_pos(model(review)[0])
            row = row._replace(**{noun_field: nouns, verb_field: verbs, adjective_field: adjectives, others_field: thers})

    except Exception as e:
        print(f"Error processing row {row}: ",e)
    return row