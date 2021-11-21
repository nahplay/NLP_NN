import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

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
        if not os.path.exists(r'results'):
            os.makedirs(r'results')
        else:
            print('The folder already exists')
        plt.savefig('results/classes_dist.png')
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
                if word not in stop_words:
                    words.append(word)
                    quantity.append(number)
            return words, quantity

        # Positive
        plt.figure(figsize=(16, 6))
        x, y = common_words(positive_counter(df))
        sns.barplot(x=x, y=y)
        plt.title('Positive words popularity')
        plt.savefig('results/positive_words.png')
        plt.show()

        # Negative
        plt.figure(figsize=(16, 6))
        x, y = common_words(negative_counter(df))
        sns.barplot(x=x, y=y)
        plt.title('Negative words popularity')
        plt.savefig('results/negative_words.png')
        plt.show()

    def stemming_wc(self, df):
        ###Stemming
        stemmer = PorterStemmer()

        def film_stemmer(row):
            text = " ".join([stemmer.stem(i) for i in row])
            return text

        df['stem_tokens'] = df['review'].map(word_tokenize)
        df['stem_tokens'] = df['stem_tokens'].map(lambda x: film_stemmer(x))
        df['stem_tokens'].head(20)

        # positive = []
        # for i in df[df.sentiment == 1].stem_tokens.str.split():
        #    for j in i:
        #        positive.append(j)

        # negative = []
        # for i in df[df.sentiment == 0].stem_tokens.str.split():
        #    for j in i:
        #        negative.append(j)

        # positive = " ".join(map(str, positive))
        # negative = " ".join(map(str, negative))

        # Positive
        # wordcloud = WordCloud(width=2000, max_words=100, height=1000, max_font_size=200).generate(positive)
        # plt.figure(figsize=(12, 10))
        # plt.imshow(wordcloud, interpolation='bilinear')
        # plt.axis("off")
        # plt.title('Positive words WORDCLOUD')
        # plt.savefig('results/wordcloud_pos.png')
        # plt.show()

        # Negative
        # wordcloud = WordCloud(width=2000, max_words=100, height=1000, max_font_size=200).generate(negative)
        # plt.figure(figsize=(12, 10))
        # plt.imshow(wordcloud, interpolation='bilinear')
        # plt.axis("off")
        # plt.title('Negative words WORDCLOUD')
        # plt.savefig('results/wordcloud_neg.png')
        # plt.show()

        return df

    def train_test_split(self, df):
        x = df['stem_tokens'].copy()
        y = df['sentiment'].copy()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

        return x_train, y_train, x_test, y_test

    def liner_model(self, x_train, y_train, x_test, y_test):
        tfidf = TfidfVectorizer(min_df=25, ngram_range=(1, 2), stop_words='english', max_features=10000,
                                sublinear_tf=True, lowercase=True)
        x_train = tfidf.fit_transform(x_train)
        x_test = tfidf.transform(x_test)
        # Defining models
        log_reg_st = LogisticRegression()  # LogReg
        sgdclass_st = SGDClassifier()  # SGD
        svm_st = LinearSVC()  # SVM

        # Hyperparams to tune
        # Log_reg
        log_reg_params = {'penalty': ['l1', 'l2'],
                          'solver': ['lbfgs', 'liblinear'],
                          'C': [0.01, 0.1, 1, 10]}
        # SGD
        sgd_params = {'penalty': ['l1', 'l2', 'elasticnet'],
                      'loss': ['log', 'modified_huber'],
                      'alpha': [0.0001, 0.01, 1, 10, 100]}
        # SVM
        svm_params = {'penalty': ['l1', 'l2'], 'loss': ['hinge', 'squared_hinge'], 'C': [0.01, 1, 10]}

        # List of models with the names
        models = [log_reg_st, sgdclass_st, svm_st]
        model_params = [log_reg_params, sgd_params, svm_params]
        modelnames = ['Logistic Regression', 'SGD', 'SVM']

        # Loop for each model
        tuned_models = []
        for model, params, modelname in zip(models, model_params, modelnames):
            print('Results for {}'.format(modelname))
            grid = GridSearchCV(estimator=model, param_grid=params, scoring='roc_auc', verbose=3, n_jobs=-1, cv=5)
            grid_result = grid.fit(x_train, y_train)
            print('Best Score for', modelname, grid_result.best_score_)
            print('Best Params for', modelname, grid_result.best_params_)
            model.set_params(**grid_result.best_params_)
            tuned_models.append(model)

            model = BaggingClassifier(base_estimator=model)

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            model.score(x_train, y_train)
            model.score(x_test, y_test)

            cm = confusion_matrix(y_test, y_pred)
            cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                     index=['Predict Positive:1', 'Predict Negative:0'])

            plt.figure(figsize=(14, 7))
            sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='icefire').set_title('Confusion matrix')

            print(classification_report(y_test, y_pred))

            cv_score = cross_val_score(model, x_train, y_train, cv=5, scoring='roc_auc')
            print('Cross-val min score', cv_score.mean())

            y_proba = model.predict_proba(x_test)[:, 1]
            print('ROC_AUC:', roc_auc_score(y_test, y_proba))

            logit_roc_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
            fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
            plt.figure(figsize=(15, 10))
            plt.plot(fpr, tpr, label='{modelname} (area = {logit_roc_auc})'.format(modelname=modelname,
                                                                                   logit_roc_auc=np.round(logit_roc_auc,
                                                                                                          decimals=2)))
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=15)
            plt.ylabel('True Positive Rate', fontsize=15)
            plt.title('Receiver operating characteristic', fontsize=15)
            plt.legend(loc="lower right", fontsize=15)
            plt.savefig('results/ROC_AUC_for_{}.png'.format(modelname))
            plt.show()
            print(150 * "-")

        plt.figure(figsize=(20, 15))
        for model, modelname in zip(tuned_models, modelnames):
            model = BaggingClassifier(base_estimator=model)
            model.fit(x_train, y_train)
            logit_roc_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
            fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:, 1])

            plt.plot(fpr, tpr, label=modelname + '(area = %0.2f)' % logit_roc_auc)

            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])

        plt.xlabel('False Positive Rate', fontsize=15)
        plt.ylabel('True Positive Rate', fontsize=15)
        plt.title('Receiver operating characteristic comparison after STEMMING')
        plt.legend(loc="lower right", fontsize=15)
        plt.savefig('results/ROC_AUC_Comparison.png')
        plt.show()

    def rnn_models(self, x_train, y_train,x_test,y_test):
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)


        vectorize_layer = TextVectorization(
            #standardize=custom_standardization,
            max_tokens=1000,
            output_mode='int',
            output_sequence_length=400)

        vectorize_layer.adapt(x_train)
        # Save model
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='results',
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         save_best_only=True)
        # Early stopping using val_loss
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


        rnn_models = []
        rnn_names= ['LSTM 2 layers', 'BLSTM 2 layers']
        model_lstm = tf.keras.Sequential([
            vectorize_layer,
            tf.keras.layers.Embedding(
                input_dim=len(vectorize_layer.get_vocabulary()),
                output_dim=64,
                mask_zero=True),
            tf.keras.layers.LSTM(64,return_sequences=True),
            tf.keras.layers.LSTM(64),
            #tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        #model_lstm.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
         #                  optimizer='adam',
         #                  metrics=[tf.keras.metrics.AUC(), 'accuracy'])#We will be checking roc_auc and accuracy
        rnn_models.append(model_lstm)

        model_blstm = tf.keras.Sequential([
            vectorize_layer,
            tf.keras.layers.Embedding(
                input_dim=len(vectorize_layer.get_vocabulary()),
                output_dim=64,
                mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            #tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        rnn_models.append(model_blstm)

        for model in rnn_models:
            model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                               optimizer='adam',
                               metrics=[tf.keras.metrics.AUC(), 'accuracy'])  # We will be checking roc_auc and accuracy
            history = model.fit(x=x_train,
                                     y=y_train,
                                     epochs=100,
                                     validation_data=(x_test, y_test),
                                     batch_size=64,
                                     callbacks=[early_stop, cp_callback]
                                     )

            # Model results visualization

            # summarize history for accuracy
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.savefig('results/accuracy.png')
            plt.show()

            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.savefig('results/loss.png')
            plt.show()
            score = model.evaluate(x_test,y_test, verbose=0)
            print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


            #Predictions
            y_pred = np.argmax(model.predict(x_test), axis=-1)
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negative', 'positive'])
            fig, ax = plt.subplots(figsize=(15, 10))
            disp.plot(ax=ax)
            plt.title('Confusion Matrix')

    def get_embs(self, link):
        path_to_downloaded_file = tf.keras.utils.get_file(
            "flower_photos",
            "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
            untar=True)


dataset = MovieDS('LargeMovieReviewDataset.csv')
prep_dataset = dataset.prep(dataset.eda())  # Storing preprocessed dataset after EDA
dataset.visualization_popular(prep_dataset)  # Visualization of popular words from preprocessed dataset
stemming_data = dataset.stemming_wc(prep_dataset)
#x_train, y_train, x_test, y_test = dataset.train_test_split(stemming_data)
#dataset.liner_model(*dataset.train_test_split(stemming_data))
dataset.rnn_models(*dataset.train_test_split(stemming_data))