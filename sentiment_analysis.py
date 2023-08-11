import csv
from transformers import AutoModel, AutoTokenizer, pipeline
import time
from multiprocessing import Pool
from collections import namedtuple, defaultdict
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def initialize_sentiment_models():
    vader_model = SentimentIntensityAnalyzer()
    nlptown = pipeline(task = "sentiment-analysis",model="nlptown/bert-base-multilingual-uncased-sentiment",top_k=None)

    model_configs = {
        'DistilBERT': ('lxyuan/distilbert-base-multilingual-cased-sentiments-student','sentiment-analysis'),
        'Finiteautomata': ('finiteautomata/bertweet-base-sentiment-analysis','sentiment-analysis'),
        'Pysentimiento': ('pysentimiento/robertuito-sentiment-analysis', 'sentiment-analysis'),
        'Cardiffnlp_Roberta': ('cardiffnlp/twitter-roberta-base-sentiment-latest', 'sentiment-analysis'),
        'Cardiffnlp_Xlm_Robert': ('cardiffnlp/twitter-xlm-roberta-base-sentiment', 'sentiment-analysis'),
        'Seethal': ('Seethal/sentiment_analysis_generic_dataset', 'sentiment-analysis'),
    }

    models = []
    models.append((vader_model,'Vader_Sentiment','Matches_Vader',None,None))

    for model_name, config in model_configs.items():
        model = pipeline(model=config[0], task=config[1], top_k=None)
        sentiment_field = f'{model_name}_Sentiment'
        matches_field = f'Matches_{model_name}'
        scores_field = f'{model_name}_Scores'
        
        models.append((model, sentiment_field, matches_field, scores_field, None))

    models.append((nlptown,'Bert_Nlptown_Sentiment','Matches_Bert_Nlptown', None,'Bert_Nlptown_Rating'))

    return models


# Standardize labels from models
def standardize_label(label):
    if label in ['positive', 'LABEL_2', 'POS']:
        return 'POS'
    if label in ['neutral', 'LABEL_1', 'NEU']:
        return 'NEU'
    if label in ['negative', 'LABEL_0', 'NEG']:
        return 'NEG'
    return 'NULL' # Possible if the model has no label. Will be counted later on to see robustness of models towards dirty data


def identify_sentiment(sentiment_output):
    max_score = 0
    max_label = ""

    for item in sentiment_output:
        if item['score'] > max_score:
            max_score = item['score']
            max_label = item['label']

    max_label = standardize_label(max_label)

    return max_label,sentiment_output


def calculate_overall_scores(row):
    # collect all model scores
    model_scores = [
        row.DistilBERT_Scores, 
        row.Finiteautomata_Scores,
        row.Pysentimiento_Scores, 
        row.Cardiffnlp_Roberta_Scores,
        row.Cardiffnlp_Xlm_Robert_Scores,
        row.Seethal_Scores
    ]
    
    # Standardize labels and compute averages
    avg_scores = {}
    total_counts = defaultdict(int)  # Required to calculate average later

    for score_list in model_scores:
        for item in score_list:
            standardized_label = standardize_label(item['label'])
            # Add up scores for averaged label
            avg_scores[standardized_label] = avg_scores.get(standardized_label, 0) + item['score']
            # Increment label count
            total_counts[standardized_label] += 1
    
    # Calculate average scores
    for label in avg_scores:
        avg_scores[label] /= total_counts[label]

    overall_sentiment = max(avg_scores, key=avg_scores.get)

    matches_Annotated_Sentiment = (overall_sentiment == row.Annotated_Sentiment)

    # Construct your new row - note namedtuple's are immutable so we're creating a new one here
    row = row._replace(
        Overall_Score=list(avg_scores.items()), 
        Overall_Sentiment=overall_sentiment, 
        Matches_Annotated=matches_Annotated_Sentiment
    )

    return row


# Assign sentiment based on polarity score. Standard values used by vader to assign sentiment
def vader_sentiment(vader_model,review):
    # The output looks like this Output: {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    if vader_model.polarity_scores(review)['compound'] >= 0.05:
        sentiment = "POS"
    elif vader_model.polarity_scores(review)['compound'] <= -0.05:
        sentiment = "NEG"
    else:
        sentiment = "NEU"

    return sentiment

# Function is for the nlptown/bert-base-multilingual-uncased-sentiment model specifically
# Documentation for the model is found here https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment
# Output of the model looks like 
# [[{'label': '5 stars', 'score': 0.6263141632080078}, {'label': '4 stars', 'score': 0.2967827618122101}, {'label': '3 stars', 'score': 0.05907348543405533}, {'label': '2 stars', 'score': 0.009653439745306969}, {'label': '1 star', 'score': 0.008176111616194248}]]
def analyze_sentiment(predictions, threshold=0.5):
    # initialize varioable
    predictions = predictions[0]
    star_rating = None
    sentiment_label = None

    # Compute for average star weight, uses probability as the weights. Something similar to GWA computation
    total_star_weight = sum(int(prediction['label'].split()[0]) * prediction['score'] for prediction in predictions)
    total_probability = sum(prediction['score'] for prediction in predictions)
    avg_star_rating = total_star_weight / total_probability  # weighted average

    # These are arbitrary values. Adjust accordingly
    if avg_star_rating < 3.1:
        sentiment_label = "NEG"
    elif avg_star_rating <= 4.1:
        sentiment_label = "NEU"
    else:
        sentiment_label = "POS"

    # Assign the star rating that has a probability score above 0.5 If there are none, assign the weighted star computation
    for prediction in predictions:
        stars = int(prediction['label'].split()[0])
        probability = prediction['score']

        if probability >= threshold:
            star_rating = stars
            break
    if star_rating is None:
        star_rating = avg_star_rating

    return sentiment_label, star_rating

def process_row(models, row):
    review = row.Review

    try:
        for model, sentiment_field, matches_field, scores, rating_field in models:
            if sentiment_field == 'Vader_Sentiment':
                # Assign the label in function created from a previous implementation
                sentiment_label = vader_sentiment(model, review)
                matches_source = (sentiment_label == row.Annotated_Sentiment)
                row = row._replace(Vader_Sentiment=sentiment_label, Matches_Vader=matches_source)

            elif sentiment_field == 'Bert_Nlptown_Sentiment':
                # this model is processed by another function
                sentiment_label, star_rating = analyze_sentiment(model(review))
                row = row._replace(Bert_Nlptown_Sentiment=sentiment_label, Matches_Bert_Nlptown=(sentiment_label == row.Annotated_Sentiment), Bert_Nlptown_Rating=star_rating)
            else:
                sentiment_label,sentiment_scores = identify_sentiment(model(review)[0]) #the output is in a list, needs to be indexed 
                matches_source = (sentiment_label == row.Annotated_Sentiment)
                row = row._replace(**{sentiment_field: sentiment_label, matches_field: matches_source,scores:sentiment_scores})

    except Exception as e:
        print(f"Error processing row {row}: ",e)
    return row