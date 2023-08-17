# Applying LLMs to NLP tasks

This repository is a proof of concept made for my Internship. There are 4 different modules, namely: sentiment analysis, topic_identification, keyword_extration, and pos_tagging.

Before anything else, all the models need to be downloaded from huggingface. Run `download.py` and all relevant models that are used in this repository will be downloaded onto your local machine. `Main.py` will perform sentiment analysis, topic identification, and keyword extraction while `pos_driver` will perform Parts of Speech Tagging on the dataset. input.csv has three columns, namely: ID (int), Review (str), and label (str). Changes to the program is necessary if it needs to be applied on a non-annotated dataset but is minor

## Sentiment analysis
There are 7 different LLMs that are used in this module alongside VADER. A traditional NLP sentiment analysis tool is being used alongside the LLMs to find a point of reference in terms of accuracy against the traditional tools. Most models simply identify the sentiment and a score; however, the model trained by NLPTown goes one step further and identifies the number of stars the user possibly would’ve given alongside the review. Another function was written that will allow this model to give a sentiment by using the number of stars.
**Input/Output:** `input.csv/sentiment.csv`

## Topic Identification
A way to identify topics using LLMs was not developed in time for the presentation; hence, traditional methods to do topic identification were used instead of LLMs, specifically: Latent Dirichlet Allocation (LDA) and Non-Negative Matrix Factorization (NMF). These algorithms are already implemented in various NLP libraries hence they were simply called and applied to the dataset. It should be noted that these algorithms look at the corpus as a whole instead of looking at it sentence per sentence. This means that the topics identified will be per batch of data fed into the algorithms. As of the moment the only pre-processing done is the removal of periods, commas, and apostrophes. It is advisable to remove stopwords and perform lemmatization for better accuracy.
**Input/Output:** `input.csv/topics.csv`

## Keyword Extraction
Three traditional keyword extraction tools and an LLM are used in this module. The traditional tools are Yake, Rake, TextRank, and KeyBert. Additionally, according to a study, a keyphrase vectorizer might improve the performance of Keybert for keyword extraction; hence, another column was added in order to allow a comparison of performance. Similar to the sentiment analysis module, each tool or model is applied to each individual review.
**Input/Output:** `input.csv/keywords.csv`

## Parts of Speech (POS) Tagging
Four POS models are used in this module. The tags used by each model may differ hence the labels must be standardized for Adjectives, Verbs, and Nouns. Other parts of speech may be added in the future. Similar to the keyword extraction and sentiment analysis modules all the models are applied to each row of review. In terms of purpose, this module is very similar to keywords but each keyword is now also sorted into its respective part of speech.
**Input/Output:** `input.csv/pos.csv`

## Initial Results
Based on a small dataset, the highest accuracy the models achieved was 77%
Upon visualizing the data from the NLPTown model, it seems that most reviews are skewed towards the positive side; hence, most of the wrongly identified reviews are identified as positive but are actually negative.
Across all NLP tasks, they struggled with the Filipino Language
More data preprocessing is advisable in order to get more accurate results. The only exception is the sentiment analysis module as some of the models are cased; however, it is important to note that a good number of reviews are in all caps. In other cultures, all caps are usually seen in a negative manner as it implies anger. Depending on the training dataset, this might be relevant to the results
Some of the models can only handle short texts; hence, some rows of the pos and sentiment analysis module are blank due to the models returning errors. It is advisable to perhaps use summarization models in the future.

## Future plans for development
- Perhaps it is better to simply fine-tune the NLPTown model to do sentiment analysis directly instead of using the function to determine sentiment
- As of the moment, there is no functionality to test the accuracy of the words extracted, topics identified, or words tagged from the pos module. A way to measure accuracy must be identified later on.
- Preprocess data for the modules topic identification, keyword extraction, and pos tagging
