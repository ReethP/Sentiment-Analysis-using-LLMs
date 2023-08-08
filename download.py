from transformers import AutoModel, AutoTokenizer

model_urls = [
    # "transformer3/H2-keywordextractor", # A possible future model to be used for keyword extraction
    "finiteautomata/bertweet-base-sentiment-analysis", # roBERTa finetuned for Sentiment Analysis using english twitter data
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student", # distilbert finetuned for Sentiment Analysis using student reviews
    "pysentimiento/robertuito-sentiment-analysis", # roBERTa finetuned for Sentiment Analysis using Spanish twitter Data
    "cardiffnlp/twitter-roberta-base-sentiment-latest", # roBERTa finetuned for Sentiment Analysis  using english twitter data
    "cardiffnlp/twitter-xlm-roberta-base-sentiment", # roBERTa finetuned on twitter data using multiple languages
    "Seethal/sentiment_analysis_generic_dataset", # BERT finetuned for Sentiment Analysis using a generic dataset
    "nlptown/bert-base-multilingual-uncased-sentiment", # BERT finetuned to predict star-rating using multilingual reviews
]

    # Other possible models to be used for text summarization / for fine-tuning
    # "google/pegasus-large",
    # "google/umt5-base",
    # "facebook/m2m100_1.2B",
    # "facebook/mbart-large-50-many-to-many-mmt",
    # "google/byt5-large",
    # "google/mt5-large"

for model_url in model_urls:
    print(f"Downloading {model_url}...")
    model = AutoModel.from_pretrained(model_url)
    # tokenizer = AutoTokenizer.from_pretrained(model_url)
    print(f"Downloaded {model_url} successfully.")