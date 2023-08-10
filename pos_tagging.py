import csv
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
from collections import namedtuple, defaultdict

# QCRI
# JJ = Adjective
# VB/VBD = Verb
# NN/NNP = Noun / Proper Noun
# some models split the tokens into half
# Kis ##met
# need to write a script that will parse this such that it will form the whole word itself

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
		
		models.append((pipeline, noun_field, verb_field, adjective_field))

	return models