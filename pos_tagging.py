import csv
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
from collections import namedtuple, defaultdict
from nltk.tokenize import TweetTokenizer

# need additional download
  # >>> import nltk
  # >>> nltk.download('omw-1.4')

# QCRI
# JJ = Adjective
# VB/VBD = Verb
# NN/NNP = Noun / Proper Noun

# TweebankNLP

# ADJ = adjective 
# VERB = verb
# ADV = adverb
# PRON = pronoun
# NOUN = noun 
# split by @

# mrm8488 tags X as an adjective


# Perform overall count here, possible add a new column for it
def overall_count(row_records):
    print("hello")

def standardize_labels(label):
    if label in ['JJ', 'ADJ','X']:
        return 'Adjective'
    if label in ['VB', 'VBD', 'VERB']:
        return 'Verb'
    if label in ['PRON', 'NOUN','PROPN', 'NN', 'NNP']:
        return 'Noun'
    return 'Others' # All other labels

def initialize_pos_models():

    model_configs = {
        'MRM_Bert': ('mrm8488/bert-spanish-cased-finetuned-pos-16-tags'),
        'Vblagoje_Bert': ('vblagoje/bert-english-uncased-finetuned-pos'),
        'QCRI_Bert': ('QCRI/bert-base-multilingual-cased-pos-english'),
        'TweebankNLP_Bert': ('TweebankNLP/bertweet-tb2_ewt-pos-tagging')
    }
    models = []
    for model_name, config in model_configs.items():
        tokenizer = AutoTokenizer.from_pretrained(config)
        model = AutoModelForTokenClassification.from_pretrained(config)
        pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
        noun_field = f'Nouns_{model_name}'
        verb_field = f'Verbs_{model_name}'
        adjective_field = f'Adjectives_{model_name}'
        others_field = f'Others_{model_name}'
        models.append((pipeline, noun_field, verb_field, adjective_field, others_field))

    return models


def identify_pos(pipeline_output):
    # Initialize a dictionary to store the last word for each type
    last_word = {'Noun': None, 'Verb': None, 'Adjective': None, 'Others': None}
    # Initialize a dictionary to store the counts for each type
    counts = {'Noun': {}, 'Verb': {}, 'Adjective': {}, 'Others': {}}
    at_flag = 0
    # Start processing the pipeline output
    for result in pipeline_output:
        # Words attached with `##` are subwords
        if '##' in result['word']:
            # Attach subword to last word based on its entity (part of speech) type
            entity_type = standardize_labels(result['entity'])
            if last_word[entity_type] is not None:
                temp = last_word[entity_type]
                last_word[entity_type] += result['word'].replace('##', '')
                counts[entity_type][last_word[entity_type]] = counts[entity_type].get(last_word[entity_type], 0) + 1
                last_word[entity_type] = None
                counts[entity_type].pop(temp)
            continue
        if '@@' in result['word']:
            # print(result)
            # Attach subword to last word based on its entity (part of speech) type
            entity_type = standardize_labels(result['entity'])
            last_word[entity_type] = result['word'].replace('@@', '')
            at_flag = 1
            continue
        if at_flag:
            # @@ is used by TweebankNLP_Bert. It behaves differently from ## as ## is placed at the beginning of the second half
            # of the word while @@ is put at the end of the first half of the word hence they need to be processed
            # differently as well
            result['word'] = last_word[entity_type] + result['word']
            at_flag = 0
        # For full words, process them based on their entity type
        entity_type = standardize_labels(result['entity'])
        last_word[entity_type] = result['word']
        counts[entity_type][last_word[entity_type]] = counts[entity_type].get(last_word[entity_type], 0) + 1

    return counts['Noun'], counts['Verb'], counts['Adjective'], counts['Others']

def process_row(models, row):
    # these are turned into # when tokenized. Simply remove them beforehand
    review = row.Review.replace(',','').replace('.','')
    # tokenizer = TweetTokenizer()
    try:
        for model, noun_field, verb_field, adjective_field, others_field in models:
            nouns, verbs, adjectives, others = identify_pos(model(review))
            row = row._replace(**{noun_field: nouns, verb_field: verbs, adjective_field: adjectives, others_field: others})
    except Exception as e:
        # print()
        print(f"Error processing row {row}: ",e)
    return row