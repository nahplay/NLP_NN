import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


class MovieDS:
    def __init__(self, path):
        self.path = path

    def eda(self):
        df = pd.read_csv(self.path)
        print(df.info)
        print(df.describe)
        # Dealing with duplicates
        print('There were', df.duplicated().sum(), 'of duplicates, which were successfully removed')
        df.drop_duplicates(inplace=True)
        ### Replace negative and positive reviews with zeroes and ones
        # Positive = 1
        # Negative = 0
        df['sentiment'] = df['sentiment'].replace(['negative'], 0)
        df['sentiment'] = df['sentiment'].replace(['positive'], 1)
        plt.hist(df[df.sentiment == 1].sentiment,
                 bins=2, color='red', label='Positive')
        plt.hist(df[df.sentiment == 0].sentiment,
                 bins=2, color='green', label='Negative')
        plt.title('Classes distribution in the train data')

        plt.xticks([])
        plt.xlim(-0.5, 2)
        plt.legend()
        plt.show()
        return df

    def prep(self, df):

        # Preprocessing is in line with https://colab.research.google.com/drive/1KUtL70jxpqglQFI_6qJcM0cM6KWix2G5?usp=sharing#scrollTo=9633a798, but with small code changes to make the filtering faster
        ### Applying lower case for our column
        df['review'] = df['review'].str.lower()
        # defining emojies
        emoj = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          u"\U00002500-\U00002BEF"  # chinese char
                          u"\U00002702-\U000027B0"
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          u"\U0001f926-\U0001f937"
                          u"\U00010000-\U0010ffff"
                          u"\u2640-\u2642"
                          u"\u2600-\u2B55"
                          u"\u200d"
                          u"\u23cf"
                          u"\u23e9"
                          u"\u231a"
                          u"\ufe0f"  # dingbats
                          u"\u3030"
                          "]+", re.UNICODE)

        df['review'] = (df['review'].map(lambda x: re.sub(r'[\w\.-]+@[\w\.-]+', '', x))
                        .map(lambda x: re.sub(r'\[[^]]*\]', "", x))
                        .map(lambda x: re.sub(r"ain't", "am not", x))
                        .map(lambda x: re.sub(r"aren't", "are not", x))
                        .map(lambda x: re.sub(r"can't", "cannot", x))
                        .map(lambda x: re.sub(r"can't've", "cannot have", x))
                        .map(lambda x: re.sub(r"cause", "because", x))
                        .map(lambda x: re.sub(r"could've", "could have", x))
                        .map(lambda x: re.sub(r"couldn't", "could not", x))
                        .map(lambda x: re.sub(r"couldn't've", "could not have", x))
                        .map(lambda x: re.sub(r"didn't", "did not", x))
                        .map(lambda x: re.sub(r"doesn't", "does not", x))
                        .map(lambda x: re.sub(r"don't", "do not", x))
                        .map(lambda x: re.sub(r"hadn't", "had not", x))
                        .map(lambda x: re.sub(r"hasn't", "has not", x))
                        .map(lambda x: re.sub(r"haven't", "have not", x))
                        .map(lambda x: re.sub(r"he'd", "he would", x))
                        .map(lambda x: re.sub(r"he'd've", "he would have", x))
                        .map(lambda x: re.sub(r"he'll", "he will", x))
                        .map(lambda x: re.sub(r"he's", "he is", x))
                        .map(lambda x: re.sub(r"how'd", "how did", x))
                        .map(lambda x: re.sub(r"how'd'y", "how do you", x))
                        .map(lambda x: re.sub(r"how'll", "how will", x))
                        .map(lambda x: re.sub(r"how's", "how is", x))
                        .map(lambda x: re.sub(r"I'd", "I would", x))
                        .map(lambda x: re.sub(r"I'd've", "I would have", x))
                        .map(lambda x: re.sub(r"I'll", "I will", x))
                        .map(lambda x: re.sub(r"I'll've", "I will have", x))
                        .map(lambda x: re.sub(r"I'm", "I am", x))
                        .map(lambda x: re.sub(r"I've", "I have", x))
                        .map(lambda x: re.sub(r"isn't", "is not", x))
                        .map(lambda x: re.sub(r"it'd", "it would", x))
                        .map(lambda x: re.sub(r"it'd've", "it would have", x))
                        .map(lambda x: re.sub(r"it'll", "it will", x))
                        .map(lambda x: re.sub(r"it'll've", "it will have", x))
                        .map(lambda x: re.sub(r"it's", "it has", x))
                        .map(lambda x: re.sub(r"let's", "let us", x))
                        .map(lambda x: re.sub(r"ma'am", "madam", x))
                        .map(lambda x: re.sub(r"mayn't", "may not", x))
                        .map(lambda x: re.sub(r"might've", "might have", x))
                        .map(lambda x: re.sub(r"mightn't", "might not", x))
                        .map(lambda x: re.sub(r"mightn't've", "might not have", x))
                        .map(lambda x: re.sub(r"must've", "must have", x))
                        .map(lambda x: re.sub(r"mustn't", "must not", x))
                        .map(lambda x: re.sub(r"mustn't've", "must not have", x))
                        .map(lambda x: re.sub(r"needn't", "need not", x))
                        .map(lambda x: re.sub(r"needn't've", "need not have", x))
                        .map(lambda x: re.sub(r"o'clock", "of the clock", x))
                        .map(lambda x: re.sub(r"oughtn't", "ought not", x))
                        .map(lambda x: re.sub(r"oughtn't've", "ought not have", x))
                        .map(lambda x: re.sub(r"shan't", "shall not", x))
                        .map(lambda x: re.sub(r"sha'n't", "shall not", x))
                        .map(lambda x: re.sub(r"shan't've", "shall not have", x))
                        .map(lambda x: re.sub(r"she'd", "she would", x))
                        .map(lambda x: re.sub(r"she'd've", "she would have", x))
                        .map(lambda x: re.sub(r"she'll", "she will", x))
                        .map(lambda x: re.sub(r"she'll've", "she will have", x))
                        .map(lambda x: re.sub(r"she's", "she is", x))
                        .map(lambda x: re.sub(r"should've", "should have", x))
                        .map(lambda x: re.sub(r"shouldn't", "should not", x))
                        .map(lambda x: re.sub(r"shouldn't've", "should not have", x))
                        .map(lambda x: re.sub(r"so've", "so have", x))
                        .map(lambda x: re.sub(r"so's", "so is", x))
                        .map(lambda x: re.sub(r"that'd", "that would", x))
                        .map(lambda x: re.sub(r"that'd've", "that would have", x))
                        .map(lambda x: re.sub(r"that's", "that is", x))
                        .map(lambda x: re.sub(r"there'd", "there would", x))
                        .map(lambda x: re.sub(r"there'd've", "there would have", x))
                        .map(lambda x: re.sub(r"there's", "there is", x))
                        .map(lambda x: re.sub(r"they'd", "they would", x))
                        .map(lambda x: re.sub(r"they'd've", "they would have", x))
                        .map(lambda x: re.sub(r"they'll", "they will", x))
                        .map(lambda x: re.sub(r"they'll've", "they will have", x))
                        .map(lambda x: re.sub(r"they're", "they are", x))
                        .map(lambda x: re.sub(r"they've", "they have", x))
                        .map(lambda x: re.sub(r"to've", "to have", x))
                        .map(lambda x: re.sub(r"wasn't", "was not", x))
                        .map(lambda x: re.sub(r"we'd", "we would", x))
                        .map(lambda x: re.sub(r"we'd've", "we would have", x))
                        .map(lambda x: re.sub(r"we'll", "we will", x))
                        .map(lambda x: re.sub(r"we'll've", "we will have", x))
                        .map(lambda x: re.sub(r"we're", "we are", x))
                        .map(lambda x: re.sub(r"we've", "we have", x))
                        .map(lambda x: re.sub(r"weren't", "were not", x))
                        .map(lambda x: re.sub(r"what'll", "what will", x))
                        .map(lambda x: re.sub(r"what'll've", "what will have", x))
                        .map(lambda x: re.sub(r"what're", "what are", x))
                        .map(lambda x: re.sub(r"what's", "what is", x))
                        .map(lambda x: re.sub(r"what've", "what have", x))
                        .map(lambda x: re.sub(r"when's", "when is", x))
                        .map(lambda x: re.sub(r"when've", "when have", x))
                        .map(lambda x: re.sub(r"where'd", "where did", x))
                        .map(lambda x: re.sub(r"where's", "where is", x))
                        .map(lambda x: re.sub(r"where've", "where have", x))
                        .map(lambda x: re.sub(r"who'll", "who will", x))
                        .map(lambda x: re.sub(r"who'll've", "who will have", x))
                        .map(lambda x: re.sub(r"who's", "who is", x))
                        .map(lambda x: re.sub(r"who've", "who have", x))
                        .map(lambda x: re.sub(r"why's", "why is", x))
                        .map(lambda x: re.sub(r"why've", "why have", x))
                        .map(lambda x: re.sub(r"won't", "will not", x))
                        .map(lambda x: re.sub(r"won't've", "will not have", x))
                        .map(lambda x: re.sub(r"would've", "would have", x))
                        .map(lambda x: re.sub(r"wouldn't", "would not", x))
                        .map(lambda x: re.sub(r"wouldn't've", "would not have", x))
                        .map(lambda x: re.sub(r"y'all", "you all", x))
                        .map(lambda x: re.sub(r"y'all'd", "you all would", x))
                        .map(lambda x: re.sub(r"y'all're", "you all are", x))
                        .map(lambda x: re.sub(r"y'all've", "you all have", x))
                        .map(lambda x: re.sub(r"you'd", "you would", x))
                        .map(lambda x: re.sub(r"you'd've", "you would have", x))
                        .map(lambda x: re.sub(r"you'll", "you will", x))
                        .map(lambda x: re.sub(r"you'll've", "you will have", x))
                        .map(lambda x: re.sub(r"you're", "you are", x))
                        .map(lambda x: re.sub(r"y'all'd've", "you all would have", x))
                        .map(lambda x: re.sub(r"you've", "you have", x))
                        .map(lambda x: re.sub(r"n\'t", " not", x))
                        .map(lambda x: re.sub(r"\'re", " are", x))
                        .map(lambda x: re.sub(r"\'s", " is", x))
                        .map(lambda x: re.sub(r"\'d", " would", x))
                        .map(lambda x: re.sub(r"\'ll", " will", x))
                        .map(lambda x: re.sub(r"\'t", " not", x))
                        .map(lambda x: re.sub(r"\'ve", " have", x))
                        .map(lambda x: re.sub(r"\'m", " am", x))
                        .map(lambda x: re.sub(' +', ' ', x))
                        .map(lambda x: re.sub(emoj, "", x))
                        .map(lambda x: re.sub(r'\[[^]]*\]', "", x))
                        .map(lambda x: re.sub(r'http(s*)?://\S+|www\.\S+', "", x))
                        .map(lambda x: re.sub(r'<.*?>', "", x))
                        )
        return df

    def visualization_popular(self, df):
        def positive_counter(df):
            positive_words = []
            for x in df[df.sentiment == 1].review.str.split():
                for i in x:
                    positive_words.append(i)

            return positive_words

        def negative_counter(df):
            negative_words = []
            for x in df[df.sentiment == 0].review.str.split():
                for i in x:
                    negative_words.append(i)

            return negative_words

        stop_words = stopwords.words("english")

        def common_words(phrase):
            counter = Counter(phrase)
            most = counter.most_common()
            words = []
            quantity = []
            for word, number in most[:50]:
                if (word not in stop_words):
                    words.append(word)
                    quantity.append(number)
            return words, quantity

        plt.figure(figsize=(16, 6))
        x, y = common_words(positive_counter(df))
        sns.barplot(x=x, y=y)
        plt.title('Positive words popularity top 50 popularity')
        plt.show()

        plt.figure(figsize=(16, 6))
        x, y = common_words(negative_counter(df))
        sns.barplot(x=x, y=y)
        plt.title('Negative words popularity')
        plt.show()


dataset = MovieDS('LargeMovieReviewDataset.csv')
prep_dataset = dataset.prep(dataset.eda())  # Storing preprocessed dataset after EDA
dataset.visualization_popular(prep_dataset)  # Visualization of popular words
