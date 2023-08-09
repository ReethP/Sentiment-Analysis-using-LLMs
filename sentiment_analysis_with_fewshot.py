import csv
from transformers import AutoModel, AutoTokenizer, pipeline
import time
from multiprocessing import Pool
from collections import namedtuple
from functools import partial
from nltk.sentiment.vader import SentimentIntensityAnalyzer

performance_metrics = {'Total_Rows':0,'Vader_Correct':0,'DistilBERT_Correct':0,'Finiteautomata_Correct':0,'Pysentimiento_Correct':0,'Cardiffnlp_Roberta_Correct':0,'Cardiffnlp_Xlm_Correct':0,'Seethal_Correct':0,'Bert_Nlptown_Correct':0}

def identify_sentiment(sentiment_output):
    max_score = 0
    max_label = ""

    for item in sentiment_output:
        if item['score'] > max_score:
            max_score = item['score']
            max_label = item['label']

    if max_label == 'positive' or max_label == 'LABEL_2' or max_label == 'POS':
        max_label = 'POS'
    elif max_label == 'neutral' or max_label == 'LABEL_1' or max_label == 'NEU':
        max_label = "NEU"
    elif max_label == 'negative' or max_label == 'LABEL_0' or max_label == 'NEG':
        max_label = "NEG"
    else:
        # Possible if the model has no label. Will be counted later on to see robustness of models towards dirty data
        max_label = "NULL"

    return max_label,sentiment_output


# def calculate_overall_scores(sentiment_output):
#     scores = [
#         row.DistilBERT_Scores,
#         row.Finiteautomata_Scores,
#         row.Pysentimiento_Scores,
#         row.Cardiffnlp_Roberta_Scores,
#         row.Cardiffnlp_Xlm_Scores,
#         row.Seethal_Scores
#     ]
#     for item in sentiment_output:
#         if item['score'] > max_score:
#             max_score = item['score']
#             max_label = item['label']

#     if max_label == 'positive' or max_label == 'LABEL_2' or max_label == 'POS':
#         max_label = 'POS'
#     elif max_label == 'neutral' or max_label == 'LABEL_1' or max_label == 'NEU':
#         max_label = "NEU"
#     elif max_label == 'negative' or max_label == 'LABEL_0' or max_label == 'NEG':
#         max_label = "NEG"
#     else:
#         # Possible if the model has no label. Will be counted later on to see robustness of models towards dirty data
#         max_label = "NULL"

#     return max_label,sentiment_output


# def calculate_overall_scores(row):
#     # Get the scores from the relevant columns
#     scores = [
#         row.DistilBERT_Scores,
#         row.Finiteautomata_Scores,
#         row.Pysentimiento_Scores,
#         row.Cardiffnlp_Roberta_Scores,
#         row.Cardiffnlp_Xlm_Scores,
#         row.Seethal_Scores
#     ]

#     # Calculate the average score
#     overall_score = sum(scores) / len(scores)

#     # Determine the overall sentiment
#     sentiments = ['positive', 'negative', 'neutral']
#     overall_sentiment = sentiments[scores.index(max(scores))]

#     # Check if overall sentiment matches Bert_Nlptown_Sentiment
#     matches_nlptown_sentiment = row.Bert_Nlptown_Sentiment == overall_sentiment

#     # Return the updated row with the calculated values
#     return row._replace(
#         Overall_Score=overall_score,
#         Overall_Sentiment=overall_sentiment,
#         Matches_Nlptown_Sentiment=matches_nlptown_sentiment
#     )

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

    # These are arbitrary values but I think it's intuitive that star values below 2.5 are negative and above 3.5 are positive
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


# Try to change pipeline to see if few-shot will improve

def process_row(models, row):
    review = row.Review

    # Since the models have many different types of outputs, they have to be processed differently.
    # TODO: Create separate functions for the other models so it looks more readable / cleaner

    try:
        for model, sentiment_field, matches_field, scores, rating_field, few_shot in models:
            if sentiment_field == 'Vader_Sentiment':
                # Assign the laber in function created from a previous implementation
                sentiment_label = vader_sentiment(model, review)
                matches_source = 'Yes' if sentiment_label == row.Annotated_Sentiment else 'No'
                row = row._replace(Vader_Sentiment=sentiment_label, Matches_Vader=matches_source)


            # elif sentiment_field == 'Seethal_Sentiment':
            #     # The Seethal_Sentiment model has a different output compared to the rest so it needs its own row processing

            #     # if few_shot:
            #     #     prompt = """ Sobrang bait ng guard and the teller was really polite. This is: LABEL_2, Everyone was rude and snappy. I expected better, sana hindi maulit. This is: LABEL_0, I have no comment, wala akong masasabi. This is: LABEL_1,"""
            #     #     review = prompt + review + ". This is: "

            #     sentiment_label,sentiment_scores = identify_sentiment(model(review)[0])
            #     matches_source = 'Yes' if sentiment_label == row.Annotated_Sentiment else 'No'
            #     row = row._replace(Seethal_Sentiment=sentiment_label, Matches_Seethal=matches_source, DistilBERT_Scores=sentiment_scores)

            elif sentiment_field == 'Bert_Nlptown_Sentiment':
                # this model is processed by another function
                sentiment_label, star_rating = analyze_sentiment(model(review))
                row = row._replace(Bert_Nlptown_Sentiment=sentiment_label, Matches_Bert_Nlptown='Yes' if sentiment_label == row.Annotated_Sentiment else 'No', Bert_Nlptown_Rating=star_rating)


            # elif sentiment_field == 'Finiteautomata_Sentiment' or sentiment_field == 'Pysentimiento_Sentiment':
            #     # if few_shot:
            #     #     prompt = """Sobrang bait ng guard and the teller was really polite. This is: POS, Everyone was rude and snappy. I expected better, sana hindi maulit. This is: NEG, I have no comment, wala akong masasabi. This is: NEU, """
            #     #     review = prompt + review + ". This is: "
            #     # sentiment_scores = model(review)[0]
            #     sentiment_label,sentiment_scores = identify_sentiment(model(review)[0])
            #     # print(sentiment_field, sentiment_scores)
            #     # sentiment_label = sentiment_scores['label']
            #     matches_source = 'Yes' if sentiment_label == row.Annotated_Sentiment else 'No'
            #     row = row._replace(**{sentiment_field: sentiment_label, matches_field: matches_source, scores:sentiment_scores})


            else:
                # All the models that have the same outputs are processed here
                # if few_shot:
                #     prompt = """Sobrang bait ng guard and the teller was really polite. This is: positive, Everyone was rude and snappy. I expected better, sana hindi maulit. This is: negative, I have no comment, wala akong masasabi. This is: neutral, """
                #     review = prompt + review + ". This is: "
                sentiment_label,sentiment_scores = identify_sentiment(model(review)[0])
                matches_source = 'Yes' if sentiment_label == row.Annotated_Sentiment else 'No'
                row = row._replace(**{sentiment_field: sentiment_label, matches_field: matches_source,scores:sentiment_scores})


    except Exception as e:
        print(f"Error processing row {row}: {model}",e)
    return row


# Define the namedtuple - header for output
Row = namedtuple('Row', [
    'Review', 
    'Annotated_Sentiment',
    # 'Is_Review_Long',
    'Vader_Sentiment', 'Matches_Vader',
    'DistilBERT_Sentiment', 'Matches_DistilBERT', 'DistilBERT_Scores',
    'Finiteautomata_Sentiment', 'Matches_Finiteautomata', 'Finiteautomata_Scores',
    'Pysentimiento_Sentiment', 'Matches_Pysentimiento', 'Pysentimiento_Scores',
    'Cardiffnlp_Roberta_Sentiment','Matches_Cardiffnlp_Roberta', 'Cardiffnlp_Roberta_Scores',
    'Cardiffnlp_Xlm_Robert_Sentiment', 'Matches_Cardiffnlp_Xlm_Robert', 'Cardiffnlp_Xlm_Scores',
    'Seethal_Sentiment', 'Matches_Seethal', 'Seethal_Scores',
    'Bert_Nlptown_Sentiment','Matches_Bert_Nlptown', 'Bert_Nlptown_Rating',
    # Requested additional column. Average the sentiment score of all columns except VADER and Bert_Nlptown. This score will then determine the sentiment
    # the Overall Score is compared again Nlptown sentiment since it has the most accurate score during preliminary testing
    'Overall_Score','Overall_Sentiment', 'Matches_Nlptown_Sentiment',
    # Adding values to namedtuple
    # Upon initial testing, few-shot really didn't improve the performance. It actually made it worse. Reason's for that may
    # be found in this study https://arxiv.org/pdf/2305.01555.pdf where more prompts or demonstrations actually made the performance worse
    # We can try different prompt designs for the model.
    # 'DistilBERT_Sentiment_FewShot', 'Matches_DistilBERT_FewShot',
    # 'Finiteautomata_Sentiment_FewShot', 'Matches_Finiteautomata_FewShot',
    # 'Pysentimiento_Sentiment_FewShot', 'Matches_Pysentimiento_FewShot',
    # 'Cardiffnlp_Roberta_Sentiment_FewShot','Matches_Cardiffnlp_Roberta_FewShot',
    # 'Cardiffnlp_Xlm_Robert_Sentiment_FewShot', 'Matches_Cardiffnlp_Xlm_Robert_FewShot',
    # 'Seethal_Sentiment_FewShot', 'Matches_Seethal_FewShot',
    ])

start_time = time.time()
vader_model = SentimentIntensityAnalyzer()
#Output: {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}

distilbert_lxyuan_model = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    top_k=None
) #Output: [{'label': 'positive', 'score': 0.518772304058075}]
#[[{'label': 'positive', 'score': 0.7131385803222656}, {'label': 'negative', 'score': 0.18599292635917664}, {'label': 'neutral', 'score': 0.10086847096681595}]]

bert_finiteautomata_model = pipeline(
    model="finiteautomata/bertweet-base-sentiment-analysis",
    top_k=None
) #Output: [{'label': 'POS', 'score': 0.5853912234306335}]
# [[{'label': 'NEU', 'score': 0.7831833362579346}, {'label': 'POS', 'score': 0.20490024983882904}, {'label': 'NEG', 'score': 0.011916464194655418}]]

robertuito_pysentimiento_model = pipeline(
    task = "sentiment-analysis",
    model="pysentimiento/robertuito-sentiment-analysis",
    top_k=None
) #Output: [{'label': 'POS', 'score': 0.5853912234306335}]
# [[{'label': 'NEU', 'score': 0.4580895006656647}, {'label': 'POS', 'score': 0.41687366366386414}, {'label': 'NEG', 'score': 0.12503689527511597}]]

roberta_cardiffnlp_model = pipeline(
    model="cardiffnlp/twitter-roberta-base-sentiment-latest", 
    tokenizer = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None
) #Output: [{'label': 'positive', 'score': 0.9225419163703918}]
# [[{'label': 'neutral', 'score': 0.5033125877380371}, {'label': 'positive', 'score': 0.4490136206150055}, {'label': 'negative', 'score': 0.04767383635044098}]]

xlm_roberta_cardiffnlp_model = pipeline(
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment", 
    tokenizer = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    top_k=None
) #Output: [{'label': 'positive', 'score': 0.5680776834487915}]
# [[{'label': 'positive', 'score': 0.5015376210212708}, {'label': 'neutral', 'score': 0.3917652368545532}, {'label': 'negative', 'score': 0.10669714957475662}]]

bert_Seethal_model = pipeline(
    model="Seethal/sentiment_analysis_generic_dataset", 
    tokenizer = "Seethal/sentiment_analysis_generic_dataset",
    top_k=None
) #Output: [{'label': 'LABEL_1', 'score': 0.49159830808639526}] where LABEL_2 is positive, LABEL_1 is neutral, and LABEL_0 is negative
# [[{'label': 'LABEL_1', 'score': 0.9817678332328796}, {'label': 'LABEL_2', 'score': 0.010082874447107315}, {'label': 'LABEL_0', 'score': 0.008149304427206516}]]

bert_nlptown_model = pipeline(
    task = "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    top_k=None
) 

models = [(vader_model, 'Vader_Sentiment', 'Matches_Vader', None ,None, False),
          (distilbert_lxyuan_model, 'DistilBERT_Sentiment', 'Matches_DistilBERT', 'DistilBERT_Scores', None, False),
          (bert_finiteautomata_model, 'Finiteautomata_Sentiment', 'Matches_Finiteautomata', 'Finiteautomata_Scores', None, False),
          (robertuito_pysentimiento_model, 'Pysentimiento_Sentiment', 'Matches_Pysentimiento', 'Pysentimiento_Scores', None, False),
          (roberta_cardiffnlp_model, 'Cardiffnlp_Roberta_Sentiment', 'Matches_Cardiffnlp_Roberta', 'Cardiffnlp_Roberta_Scores', None, False),
          (xlm_roberta_cardiffnlp_model, 'Cardiffnlp_Xlm_Robert_Sentiment', 'Matches_Cardiffnlp_Xlm_Robert', 'Cardiffnlp_Xlm_Scores',None, False),
          (bert_Seethal_model, 'Seethal_Sentiment', 'Matches_Seethal', 'Seethal_Scores',None, False),
          (bert_nlptown_model, 'Bert_Nlptown_Sentiment', 'Matches_Bert_Nlptown', 'Bert_Nlptown_Rating', None, False)
          # Repeating the models for few-shot prompting. VADER and bert-nlptown is not added as VADER is not an LLM and bert-nlptown outputs 
          # number of stars the review may leave. It already performs rather well so I'm going to leave it alone
          # ,(distilbert_lxyuan_model, 'DistilBERT_Sentiment_FewShot', 'Matches_DistilBERT_FewShot', None, True),
          # (bert_finiteautomata_model, 'Finiteautomata_Sentiment_FewShot', 'Matches_Finiteautomata_FewShot', None, True),
          # (robertuito_pysentimiento_model, 'Pysentimiento_Sentiment_FewShot', 'Matches_Pysentimiento_FewShot', None, True),
          # (roberta_cardiffnlp_model, 'Cardiffnlp_Roberta_Sentiment_FewShot', 'Matches_Cardiffnlp_Roberta_FewShot', None, True),
          # (xlm_roberta_cardiffnlp_model, 'Cardiffnlp_Xlm_Robert_Sentiment_FewShot', 'Matches_Cardiffnlp_Xlm_Robert_FewShot', None, True),
          # (bert_Seethal_model, 'Seethal_Sentiment_FewShot', 'Matches_Seethal_FewShot', None, True)
          ]

# prompt = """Sobrang bait ng guard and the teller was really polite. This is: positive, Everyone was rude and snappy. I expected better, sana hindi maulit. This is: negative, I have no comment, wala akong masasabi. This is: neutral, """
# The above prompt is demonstrative, it decreased the performance of the model. Leaning more towards the negative side it seems. It's possible that there's a bug so need to investigate

with open('input.csv', 'r') as file, open('output.csv', 'w', newline='') as outfile:
    start_time = time.time()
    reader = csv.reader(file)
    writer = csv.writer(outfile)

    # Writing the header of the CSV
    writer.writerow(list(Row._fields))

    # Populate the first two columns Review and Annotated_Sentiment with data from the file
    row_records = [Row(row[0], row[1], *[''] * (len(Row._fields) - 2)) for row in reader]

    # Process data. partial() makes it easier to use the many different models to use the same function
    for model in models:
        process_row_with_model = partial(process_row, [model])
        row_records = (map(process_row_with_model, row_records))

    # row_records = map(calculate_overall_scores, row_records)

    writer.writerows(row_records)  # Write all the modified rows to the output CSV

end_time = time.time()
print("Execution Time: ", end_time-start_time, " seconds")



