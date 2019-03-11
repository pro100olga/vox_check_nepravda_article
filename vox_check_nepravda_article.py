import pandas as pd
import numpy as np

# =============================================================================
# # Import data from csv
# 
# =============================================================================

pd.set_option('display.max_column', None)

d = pd.read_csv('vox_lies.csv')

d = d[d.columns[[1,5,6]]]

d = d.rename(index = str, columns={'Особа': 'person', 'Цитата': 'citation', 'Оцінка VoxCheck': 'vox_evaluation'})

d = d[~d['person'].isnull()]

d.head()

# =============================================================================
# # Remove speakers with low number of citation
# =============================================================================

d['person'].value_counts()

d = d[~d['person'].isin(['Володимир Зеленський', 'Андрій Садовий', 'Олег Тягнибок', 'Юрій Бойко'])]
d_wo_rabinovich = d[~d['person'].isin(['Володимир Зеленський', 'Андрій Садовий', 'Олег Тягнибок', 'Юрій Бойко', 'Вадим Рабінович'])]

d = d.reset_index()
d_wo_rabinovich = d_wo_rabinovich.reset_index()

# =============================================================================
# TF IDF
# =============================================================================

import string
from nltk.tokenize import word_tokenize
from gensim.corpora.dictionary import Dictionary as gensim_dict
from gensim.models.tfidfmodel import TfidfModel
from nltk.stem.snowball import SnowballStemmer
from stop_words import get_stop_words
import re


# Functions

def remove_punctuation_from_text(text):
    table = str.maketrans({key: None for key in string.punctuation})
    text_wo_punct = text.translate(table) 
    text_wo_punct = re.sub(pattern = "—|»|«", repl = "", string = text_wo_punct)
    text_wo_punct = re.sub(pattern = "\n", repl = " ", string = text_wo_punct)
    text_wo_punct = re.sub(pattern = " +", repl = " ", string = text_wo_punct)
    return text_wo_punct

def create_corpus_from_list_with_texts_incl_stemming(texts_texts_list): 
    
    # takes a list of texts, makes it lowercase, tokenizes, stemms, creates corpus and dictionary
    
    # Lowercase
    
    texts_texts_list = [t.lower() for t in texts_texts_list]
    
    # Remove punctuation 
    
    texts_texts_list = [remove_punctuation_from_text(t) for t in texts_texts_list]
    
    # Tokenize documents

    texts_texts_list_tokenized_all_words = [word_tokenize(t) for t in texts_texts_list]
    
    # Leave only words

    texts_texts_tokenized = []
    
    for i in range(0,len(texts_texts_list_tokenized_all_words)):
        texts_texts_tokenized.append([w for w in texts_texts_list_tokenized_all_words[i] if w.isalpha()])
    
    del texts_texts_list_tokenized_all_words
    
    # Remove all one-letter words (some names in plays are written as 'o l g a')

    texts_texts_tokenized_with_one_letter_words = texts_texts_tokenized
    texts_texts_tokenized = []
     
    for i in range(0,len(texts_texts_tokenized_with_one_letter_words)):
        texts_texts_tokenized.append([w for w in texts_texts_tokenized_with_one_letter_words[i] if len(w) > 1])
    
    del texts_texts_tokenized_with_one_letter_words
    
    # Stemming
    
    ru_stemmer = SnowballStemmer("russian")
    
    for i in range(0,len(texts_texts_tokenized)):
        texts_texts_tokenized[i] = [ru_stemmer.stem(w) for w in texts_texts_tokenized[i]]
    
    
    # Create and return corpus and dictionary

    texts_texts_tokenized_dict = gensim_dict(texts_texts_tokenized)
    
    res_corpus = [texts_texts_tokenized_dict.doc2bow(t) for t in texts_texts_tokenized]
    
    return res_corpus, texts_texts_tokenized_dict


def calculate_top_n_important_words_from_corpus_and_dict(texts_corpus, texts_dict,n):
    
    # from corpus and dictionary creates and print top N words
    
    tfidf = TfidfModel(texts_corpus)

    for doc in texts_corpus:
    
        tfidf_weights = tfidf[doc]
        
        # Sort the weights from highest to lowest: sorted_tfidf_weights
        sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)
        
        # Print the top  weighted words
        for term_id, weight in sorted_tfidf_weights[:n]:
            print(texts_dict.get(term_id), weight)
        
        print("\n")


# Convert dataframe with citations to list for functions
        
texts = {}

stop_words = list(set(get_stop_words('russian') + get_stop_words('ukrainian')))

stop_words = [re.sub(pattern = "\\'", repl = "", string = w) for w in stop_words]
stop_words = [re.sub(pattern = '\\"', repl = '', string = w) for w in stop_words]
stop_words = [re.sub(pattern = '\\“|\\”', repl = '', string = w) for w in stop_words]


for index, row in d.iterrows():
    text = d.iloc[index]['citation']
    text = text.lower()
    text = remove_punctuation_from_text(text)
    text = re.sub(pattern = "\\'", repl = "", string = text)
    text = re.sub(pattern = '\\"', repl = '', string = text)
    text = re.sub(pattern = '\\“|\\”', repl = '', string = text)
    for stop_word in stop_words:
        text = re.sub(pattern = r"\b" + stop_word + "\\b", repl = '', string = text)
    if d.iloc[index]['person'] in texts.keys():
        texts[d.iloc[index]['person']] = texts[d.iloc[index]['person']] + ' ' + text
    else:
        texts[d.iloc[index]['person']] = text

del index, row, stop_word, text  

texts_corpus_incl_stem, texts_dict_incl_stem = create_corpus_from_list_with_texts_incl_stemming(list(texts.values()))
calculate_top_n_important_words_from_corpus_and_dict(texts_corpus_incl_stem, texts_dict_incl_stem, 10)

# =============================================================================
# # Predicting author
# =============================================================================

# Split into train and test

from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(d['citation'], 
                                                                    d['person'], 
                                                                    random_state = 10,
                                                                    test_size = 0.2,
                                                                    stratify = d['person'])

# Count Vectorizer

from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words

stop_words = list(set(get_stop_words('russian') + get_stop_words('ukrainian')))

vect = CountVectorizer(stop_words = stop_words, ngram_range = (1,2)).fit(train_data)

train_data_vectorized = vect.transform(train_data)
test_data_vectorized = vect.transform(test_data)

# Create Naive Bayes Classifier

from sklearn import naive_bayes

model_nb = naive_bayes.MultinomialNB()
model_nb.fit(train_data_vectorized, train_labels)


from sklearn import model_selection

predicted_labels_train_nb_cv = model_selection.cross_val_predict(model_nb, train_data_vectorized, train_labels, cv = 5)

pd.crosstab(train_labels, predicted_labels_train_nb_cv)

print(pd.crosstab(test_labels, model_nb.predict(test_data_vectorized)))

print('train: ' + str(sum(train_labels == predicted_labels_train_nb_cv)/len(train_labels)))
print('test: ' + str(sum(test_labels == model_nb.predict(test_data_vectorized))/len(test_labels)))

feature_names = np.array(vect.get_feature_names())

for i in range(0, len(model_nb.classes_)):
    print(model_nb.classes_[i])
    sorted_coef_index = model_nb.coef_[i].argsort()
    # print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:20]]))
    print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

# Check misclassified citations
    
test_real_and_predicted = pd.concat([pd.DataFrame(test_data).reset_index(),
                                     pd.DataFrame(test_labels).reset_index(),
                                     pd.DataFrame(model_nb.predict(test_data_vectorized)).reset_index()],
    axis = 1)

test_real_and_predicted = test_real_and_predicted[['citation', 'person', 0]]

test_real_and_predicted = test_real_and_predicted.rename(index = str, 
                                                         columns={'person': 'person_real', 0: 'person_predicted'})

test_real_and_predicted[(test_real_and_predicted['person_real'] == 'Анатолій Гриценко') & 
                        (test_real_and_predicted['person_predicted'] == 'Олег Ляшко')]['citation'][5]

del i, sorted_coef_index, train_data, train_labels, test_data, test_labels, feature_names, predicted_labels_train_nb_cv

# =============================================================================
# # Predicting author without Rabinovich
# =============================================================================

# Split into train and test

from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(d_wo_rabinovich['citation'], 
                                                                    d_wo_rabinovich['person'], 
                                                                    random_state = 10,
                                                                    test_size = 0.2,
                                                                    stratify = d_wo_rabinovich['person'])

# Count Vectorizer

from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words

stop_words = list(set(get_stop_words('russian') + get_stop_words('ukrainian')))

vect = CountVectorizer(stop_words = stop_words, ngram_range = (1,2)).fit(train_data)

train_data_vectorized = vect.transform(train_data)
test_data_vectorized = vect.transform(test_data)

# Create Naive Bayes Classifier

from sklearn import naive_bayes

model_nb = naive_bayes.MultinomialNB()
model_nb.fit(train_data_vectorized, train_labels)


from sklearn import model_selection

predicted_labels_train_nb_cv = model_selection.cross_val_predict(model_nb, train_data_vectorized, train_labels, cv = 5)

pd.crosstab(train_labels, predicted_labels_train_nb_cv)

print(pd.crosstab(test_labels, model_nb.predict(test_data_vectorized)))

print('train: ' + str(sum(train_labels == predicted_labels_train_nb_cv)/len(train_labels)))
print('test: ' + str(sum(test_labels == model_nb.predict(test_data_vectorized))/len(test_labels)))

feature_names = np.array(vect.get_feature_names())

for i in range(0, len(model_nb.classes_)):
    print(model_nb.classes_[i])
    sorted_coef_index = model_nb.coef_[i].argsort()
    # print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:20]]))
    print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

del i, sorted_coef_index, train_data, train_labels, test_data, test_labels, feature_names, predicted_labels_train_nb_cv
