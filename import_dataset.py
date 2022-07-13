from nbformat import write
import streamlit as st
import mysql.connector
import tweepy
import config
import pandas as pd
import numpy as np
import nltk
import enchant
import re
import string
from pandas import DataFrame
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import xlrd
from sklearn.model_selection import train_test_split
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import time


# Import dataset
book = xlrd.open_workbook("dataseet/dataset_final2.xls")
sheet = book.sheet_by_name('Sheet1')
en = enchant.Dict("en_US")
idn = []
with open('dataseet/wordlist-id.txt', 'r') as file:
    for word in file:
        idn.append(word)

for n in range(1, sheet.nrows):
    text = sheet.cell(n, 0).value
    ext = sheet.cell(n, 1).value
    neu = sheet.cell(n, 2).value
    agr = sheet.cell(n, 3).value
    con = sheet.cell(n, 4).value
    opn = sheet.cell(n, 5).value
    dokumen = text
    dokumen = dokumen.lower()
    dokumen = re.sub(r'@[^\s]+', '', dokumen)
    dokumen = re.sub(r'#([^\s]+)', '', dokumen)
    dokumen = re.sub(r'https:[^\s]+', '', dokumen)
    dokumen = dokumen.translate(
        str.maketrans("", "", string.punctuation))
    dokumen = re.sub(r'[^\x00-\x7f]+', '', dokumen)
    dokumen = re.sub(r'\s+', ' ', dokumen)
    dokumen = re.sub(r"\d+", "", dokumen)
    tokens = nltk.tokenize.word_tokenize(dokumen)
    tokens = str(tokens)
    factorySt = StemmerFactory()
    stemmer = factorySt.create_stemmer()
    hasil_stemming = stemmer.stem(tokens)
    factorySw = StopWordRemoverFactory()
    stopword = factorySw.create_stop_word_remover()
    hasil_stopword_removal = stopword.remove(hasil_stemming)
    slangwords = dict()
    with open('dataseet/slangword-id.txt') as wordfile:
        for word in wordfile:
            word = word.split('=')
            slangwords[word[0]] = word[1].replace('\n', '')
            wordsArray, fixed = hasil_stopword_removal.split(' '), []
    for word in wordsArray:
        if word in slangwords:
            word = slangwords[word]
        fixed.append(word)
        hasil_slang_word = ' '.join(fixed)
    # Hapus character yang berulang
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    hps_loop_char = ''
    for word in hasil_slang_word.split(' '):
        if word != '':
            if en.check(word):
                hps_loop_char += word+' '
            elif word in idn:
                hps_loop_char += word+' '
            else:
                hps_loop_char += pattern.sub(r"\1", word) + ' '

    values = (hps_loop_char, ext, neu, agr, con, opn)

    db = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="klasifikasi"
    )

    cursor = db.cursor()
    sql = "INSERT INTO ciri_kepribadian (text, ext, neu, agr, con, opn) VALUES (%s, %s, %s, %s, %s, %s)"
    val = values
    cursor.execute(sql, val)
    db.commit()

st.write("{} dataset ditambahkan".format(cursor.rowcount))
