from transformers import AutoModel, AutoTokenizer

model_urls = [
    "transformer3/H2-keywordextractor"
    # "google/pegasus-large",
    # "google/umt5-base",
    # "facebook/m2m100_1.2B",
    # "facebook/mbart-large-50-many-to-many-mmt",
    # "google/byt5-large",
    # "google/mt5-large"
    # "finiteautomata/bertweet-base-sentiment-analysis",
    # "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    # "pysentimiento/robertuito-sentiment-analysis",
    # "GKLMIP/roberta-tagalog-base",
    # "jcblaise/roberta-tagalog-base",
    # "xlm-roberta-base",
    # "roberta-base",
    # "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli",
    # "cardiffnlp/twitter-xlm-roberta-base",
    # "cardiffnlp/twitter-roberta-base-sentiment-latest",
    # "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    # "Seethal/sentiment_analysis_generic_dataset",
    # "nlptown/bert-base-multilingual-uncased-sentiment",
    # "deepset/roberta-base-squad2",
    # "xlm-roberta-large-finetuned-conll03-english",
    # "papluca/xlm-roberta-base-language-detection",
    # "deepset/xlm-roberta-large-squad2",
    # "timpal0l/mdeberta-v3-base-squad2",
    # "RashidNLP/Finance-Sentiment-Classification",
    # "xlm-roberta-large",
    # "joeddav/xlm-roberta-large-xnli",
    # "roberta-large-mnli",
    # "cardiffnlp/twitter-roberta-base-irony",
    # "bert-large-uncased"
]
    # "cardiffnlp/twitter-xlm-roberta-base-sentiment",

for model_url in model_urls:
    print(f"Downloading {model_url}...")
    model = AutoModel.from_pretrained(model_url)
    # tokenizer = AutoTokenizer.from_pretrained(model_url)
    print(f"Downloaded {model_url} successfully.")


    # "finiteautomata/bertweet-base-sentiment-analysis", - READY TO USE Sentiment Analysis
    # "lxyuan/distilbert-base-multilingual-cased-sentiments-student", - READY TO USE Sentiment Analysis
    # "pysentimiento/robertuito-sentiment-analysis", - READY TO USE Sentiment Analysis FOR SPANISH ONLY
    # "cardiffnlp/twitter-roberta-base-sentiment-latest", - READY TO USE Sentiment ANalysis
    # "cardiffnlp/twitter-xlm-roberta-base-sentiment", - READY TO USE Sentiment Analysis - multilingual
    # "Seethal/sentiment_analysis_generic_dataset", - Ready to use Sentiment Analysis
    
    # "joeddav/xlm-roberta-large-xnli" - READY TO USE Zero-shot classification MULTILINGUAL

    # "cardiffnlp/twitter-xlm-roberta-base-sentiment", #- READY TO USE Sentiment Analysis - multilingual 

    # "google/umt5-base", - needs fine-tuning
    # "facebook/m2m100_1.2B", - Purpose: translator
    # "facebook/mbart-large-50-many-to-many-mmt", Purpose: translator
    # "google/byt5-large", - unsure
    # "google/mt5-large" - unsure
    # "finiteautomata/bertweet-base-sentiment-analysis", - READY TO USE Sentiment Analysis
    # "lxyuan/distilbert-base-multilingual-cased-sentiments-student", - READY TO USE Sentiment Analysis
    # "pysentimiento/robertuito-sentiment-analysis", - READY TO USE Sentiment Analysis FOR SPANISH ONLY
    # "GKLMIP/roberta-tagalog-base", NEEDS PAPER CHECKING https://github.com/GKLMIP/Pretrained-Models-For-Tagalog/blob/main/README.md
    # "jcblaise/roberta-tagalog-base", - Fill mask, needs fine-tuning
    # "xlm-roberta-base",
    # "roberta-base",
    # "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli", - no clue what its for and not useful I think
    # "cardiffnlp/twitter-xlm-roberta-base", - masked 
    # "cardiffnlp/twitter-roberta-base-sentiment-latest", - READY TO USE Sentiment ANalysis
    # "cardiffnlp/twitter-xlm-roberta-base-sentiment", - READY TO USE Sentiment Analysis - multilingual
    # "Seethal/sentiment_analysis_generic_dataset", - Ready to use Sentiment Analysis
    # "nlptown/bert-base-multilingual-uncased-sentiment", - Predicts stars according to sentiment. Ready to use but needs finetuning for tasks
    # "joeddav/xlm-roberta-large-xnli" - READY TO USE Zero-shot classification MULTILINGUAL