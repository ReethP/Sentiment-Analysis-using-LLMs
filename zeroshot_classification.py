import csv
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
from collections import namedtuple, defaultdict

Zero_Shot_Header = namedtuple('Row', [
    'ID', 'Review', 'Annotated_Sentiment',
    'Labels_MoritzMNLI_Deberta',
    'Labels_MoritzMulti_Deberta',
    'Labels_Crossencoder_Roberta',
    'Labels_Valhalla_Distilbart',
    'Labels_Narsil_Deberta',
    'Labels_Crossencoder_Deberta'
])



def initialize_zeroshot_models():

	model_configs = {
		'MoritzMNLI_Deberta': ('MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'),
		'MoritzMulti_Deberta': ('MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'),
		'Crossencoder_Roberta': ('cross-encoder/nli-deberta-base'),
		'Valhalla_Distilbart': ('valhalla/distilbart-mnli-12-1'),
		'Narsil_Deberta': ('Narsil/deberta-large-mnli-zero-cls'),
		'Crossencoder_Deberta': ('cross-encoder')
	}

	models = []

	# for model_name, config in model_configs.items():
	# 	tokenizer = AutoTokenizer.from_pretrained(config[0])
	# 	model = AutoModelForTokenClassification.from_pretrained(config[0])
	# 	pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
	# 	noun_field = f'Nouns_{model_name}'
	# 	verb_field = f'Verbs_{model_name}'
	# 	adjective_field = f'Adjectives_{model_name}'
		
	# 	models.append((pipeline, noun_field, verb_field, adjective_field))

	return models