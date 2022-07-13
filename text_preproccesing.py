import csv
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import WhitespaceTokenizer
import streamlit as st
import pandas as pd
import nltk
import xlrd
import string
string.punctuation

book = xlrd.open_workbook("dataseet/essaytrain_1.xls")
sheet = book.sheet_by_name('essaytrain')

dokumen = []
for n in range(1, sheet.nrows):
    text = sheet.cell(n, 1).value
    dokumen.append(text)
data = pd.DataFrame({'text': dokumen})

# defining the function to remove punctuation


def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree


# storing the puntuation free text
data['hasil'] = data['text'].apply(lambda x: remove_punctuation(x))
data['lower'] = data['hasil'].apply(lambda x: x.lower())


# defining function for tokenization
tk = WhitespaceTokenizer()


def tokenization(text):
    # tokens = re.split('W+', text)
    tokens = tk.tokenize(text)
    return tokens


# applying function to the column
data['tokenied'] = data['lower'].apply(lambda x: tokenization(x))

stopwords = nltk.corpus.stopwords.words('english')


def remove_stopwords(text):
    output = [i for i in text if i not in stopwords]
    return output


data['no_stopwords'] = data['tokenied'].apply(lambda x: remove_stopwords(x))

porter_stemmer = PorterStemmer()


def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text


data['msg_stemmed'] = data['no_stopwords'].apply(lambda x: stemming(x))

wordnet_lemmatizer = WordNetLemmatizer()


def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text


data['msg_lemmatized'] = data['no_stopwords'].apply(lambda x: lemmatizer(x))

st.write(data.head())

data['msg_lemmatized'].to_csv('dataseet/hasil_preproccesing.csv')

# tentukan lokasi file, nama file, dan inisialisasi csv

# f = open('dataseet/hasil_preproccesing.csv', 'w')
# w = csv.writer(f)
# w.writerow(('Text'))

# # menulis file csv
# for s in data['msg_lemmatized']:
#     w.writerow(s)


# # menutup file csv

# f.close()
