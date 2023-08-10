from transformers import AutoModel, AutoTokenizer

model_urls = [

    # Sentiment Analysis models
    # "transformer3/H2-keywordextractor", # A possible future model to be used for keyword extraction
    # "finiteautomata/bertweet-base-sentiment-analysis", # roBERTa finetuned for Sentiment Analysis using english twitter data
    # "lxyuan/distilbert-base-multilingual-cased-sentiments-student", # distilbert finetuned for Sentiment Analysis using student reviews
    # "pysentimiento/robertuito-sentiment-analysis", # roBERTa finetuned for Sentiment Analysis using Spanish twitter Data
    # "cardiffnlp/twitter-roberta-base-sentiment-latest", # roBERTa finetuned for Sentiment Analysis  using english twitter data
    # "cardiffnlp/twitter-xlm-roberta-base-sentiment", # roBERTa finetuned on twitter data using multiple languages
    # "Seethal/sentiment_analysis_generic_dataset", # BERT finetuned for Sentiment Analysis using a generic dataset
    # "nlptown/bert-base-multilingual-uncased-sentiment", # BERT finetuned to predict star-rating using multilingual reviews

    # POS Tagging Models
    # 'mrm8488/bert-spanish-cased-finetuned-pos-16-tags',
    # 'vblagoje/bert-english-uncased-finetuned-pos',
    # 'QCRI/bert-base-multilingual-cased-pos-english',
    # 'TweebankNLP/bertweet-tb2_ewt-pos-tagging',

    # Zero-Shot classification

    # 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
    # 'facebook/bart-large-mnli',
    
    # 'joeddav/xlm-roberta-large-xnli',
    # 'vicgalle/xlm-roberta-large-xnli-anli',
    # 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli',
    # 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',
    # 'cross-encoder/nli-deberta-base',
    # 'valhalla/distilbart-mnli-12-1',
    # 'Narsil/deberta-large-mnli-zero-cls',
    # 'cross-encoder/nli-deberta-base'

    # https://huggingface.co/joeddav/xlm-roberta-large-xnli too big for machine 
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