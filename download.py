from transformers import AutoModel, AutoTokenizer

model_urls = [
    # "transformer3/H2-keywordextractor", # possible future model to be used for keyword extraction
    "finiteautomata/bertweet-base-sentiment-analysis", # Sentiment Analysis roBERTa finetuned on english twitter data
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student", # distilbert Sentiment Analysis
    "pysentimiento/robertuito-sentiment-analysis", # Sentiment Analysis roBERTa finetuned on Spanish twitter Data
    "cardiffnlp/twitter-roberta-base-sentiment-latest", # Sentiment Analysis roBERTa finetuned on english twitter data
    "cardiffnlp/twitter-xlm-roberta-base-sentiment", # roBERTa finetuned on english twitter data finetuned on multiple languages
    "Seethal/sentiment_analysis_generic_dataset", # Ready to use Sentiment Analysis BERT
    "nlptown/bert-base-multilingual-uncased-sentiment", # Ready to use star-rating prediction BERT finetuned on multilingual reviews
]

    # Other possibble models to be used for text summarization / for fine-tuning
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