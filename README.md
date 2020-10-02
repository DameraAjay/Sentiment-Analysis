# Sentiment-Analysis
    Sentiment Analysis on movie reviews using Logistic regression and tf-idf vectors.
    Sentiment analysis developed by considering 7000 (3500 positive and 3500 negative) train samples and 1000 test samples from aclimdb dataset.
    Download IMDb dataset from https://www.imdb.com/interfaces/
# step1:
    preprocessing -> removed stopwords, punctuations and applied lemmatization
# step2:
    converting each review to tf-idf vector using sklearn
# step3:
    applying logistic regression on tf-idf vectors
# step4:
    calculating accuracy 
    Accuracy: 0.758
