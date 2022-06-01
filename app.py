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
import seaborn as sns
from pandas import DataFrame
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import xlrd
from sklearn.model_selection import train_test_split
import time
import hydralit_components as hc

# Page Config
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
    page_title="Klasifikasi Kepribadian",
    page_icon="images/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# Auth API
api_key = config.API_KEY
api_key_secret = config.API_SECRET
access_token = config.ACCESS_TOKEN
access_token_secret = config.ACCESS_TOKEN_SECRET
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Sidebar
st.sidebar.subheader("Input data")
nama_pengguna = st.sidebar.text_input(
    "Username Twitter", placeholder="masukkan username")
st.sidebar.markdown("_Enter_ _untuk_ _apply_")

# Canvas Area
if nama_pengguna == "":
    with hc.HyLoader('loading', hc.Loaders.standard_loaders, index=[3]):
        time.sleep(3)
if nama_pengguna != "":
    with hc.HyLoader('loading', hc.Loaders.standard_loaders, index=[3]):
        time.sleep(3)


st.title("Sistem Klasifikasi Kepribadian Berdasarkan Postingan di Twitter")


def load_data():
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="klasifikasi"
    )
    cursor = db.cursor()
    sql = "SELECT * FROM ciri_kepribadian"
    cursor.execute(sql)
    results = cursor.fetchall()
    rows = []
    index = []
    for data in results:
        i = 1
        rows.append({'text': data[1], 'label': data[2]})
        index.append(data[0])
    return DataFrame(rows, index=index)


def klasifikasi(a):
    example_counts = vectorizer.transform(a)
    prediksi = classifier.predict(example_counts)
    return prediksi


data = DataFrame({'text': [], 'label': []})
data = data.append(load_data())

vectorize = CountVectorizer()
classifier = MultinomialNB()
vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, norm=None, smooth_idf=True)
hitung = vectorizer.fit_transform(data['text'].values)
# hitung = vectorize.fit_transform(data['text'].values)
# st.write(hitung.toarray())
classifier = MultinomialNB()
target = data['label'].values
classifier.fit(hitung, target)


# ____________________________________________________________________________________________________________


en = enchant.Dict("en_US")
idn = []
with open('dataseet/wordlist-id.txt', 'r') as file:
    for word in file:
        idn.append(word)


def text_proccesing(dokumen):
    # Case Folding
    lower_case = dokumen.lower()

    # Tokenizing
    # Username removal
    lower_case = re.sub(r'@[^\s]+', '', lower_case)

    # Hastag Removal
    lower_case = re.sub(r'#([^\s]+)', '', lower_case)

    # URL removal
    lower_case = re.sub(r'https:[^\s]+', '', lower_case)

    # Symbol removal
    lower_case = lower_case.translate(
        str.maketrans("", "", string.punctuation))

    # ASCII chars
    lower_case = re.sub(r'[^\x00-\x7f]+', '', lower_case)

    # Double spasi
    lower_case = re.sub(r'\s+', ' ', lower_case)

    # number removal
    lower_case = re.sub(r"\d+", "", lower_case)

    # Token
    tokens = nltk.tokenize.word_tokenize(lower_case)
    freq_tokens = nltk.FreqDist(tokens)
    freq_tokens.plot(30, cumulative=False)
    grafik = plt.show()

    # Stemming
    token = str(tokens)
    factorySt = StemmerFactory()
    stemmer = factorySt.create_stemmer()
    hasil_stemming = stemmer.stem(token)

    # Stopword Removal
    factorySw = StopWordRemoverFactory()
    stopword = factorySw.create_stop_word_remover()
    hasil_stopword_removal = stopword.remove(hasil_stemming)

    # Slang-word converting
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
    # Connect Db
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="klasifikasi"
    )
    cursor = db.cursor()
    sql = "INSERT INTO data_pengguna (username, tweet) VALUES (%s, %s)"
    val = (nama_pengguna, hps_loop_char)
    cursor.execute(sql, val)
    db.commit()
    # st.markdown("_Akun ditemukan_".format(cursor.rowcount))


def run_klasifikasi():
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="klasifikasi"
    )
    cursor = db.cursor()
    sql = """SELECT * FROM data_pengguna WHERE username = %s ORDER BY id DESC"""
    dat = []
    val = nama_pengguna
    cursor.execute(sql, (val,))
    results = cursor.fetchall()
    # for data in results:
    #     dat.append(data[2])
    dat.append(results[0][2])
    # dat.array.reshape(-1, 1)
    b = klasifikasi(dat)
    st.markdown('**Hasil Klasifikasi Kepribadian :**')
    if b == 'Neuroticism':
        st.markdown(
            """<p style="font-size:medium;color:#e74c3c;"/p><b>Neuroticism</b>""", unsafe_allow_html=True)
        st.write("Neuroticism adalah dimensi kepribadian yang menilai kemampuan seseorang dalam menahan tekanan atau stress. Karakteristik Positif dari Neuroticism disebut dengan Emotional Stability (Stabilitas Emosional), Individu dengan Emosional yang stabil cenderang Tenang saat menghadapi masalah, percaya diri, memiliki pendirian yang teguh.")
        st.write("Sedangkan karakteristik kepribadian Neuroticism (karakteristik Negatif) adalah mudah gugup, depresi, tidak percaya diri dan mudah berubah pikiran.")
        st.write("Oleh karena itu, Dimensi Kepribadian Neuroticism atau Neurotisme yang pada dasarnya merupakan sisi negatif ini  sering disebut juga dengan dimensi Emotional Stability (Stabilitas Emosional) sebagai sisi positifnya, ada juga yang menyebut Dimensi ini sebagai Natural Reactions (Reaksi Alami).")
        st.markdown(
            """<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    elif b == 'Agreeableness':
        st.markdown(
            """<p style="font-size:medium;color:#e74c3c;"/p><b>Agreeableness</b>""", unsafe_allow_html=True)
        st.write("Individu yang berdimensi Agreableness ini cenderung lebih patuh dengan individu lainnya dan memiliki kepribadian yang ingin menghindari konfilk. Karakteristik Positif-nya adalah kooperatif (dapat bekerjasama), penuh kepercayaan, bersifat baik, hangat dan berhati lembut serta suka membantu.")
        st.write("Karakteristik kebalikan dari sifat “Agreeableness” adalah mereka yang tidak mudah bersepakat dengan individu lain karena suka menentang, bersifat dingin dan tidak ramah.")
        st.markdown(
            """<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    elif b == 'Conscientiousness':
        st.markdown(
            """<p style="font-size:medium;color:#e74c3c;"/p><b>Conscientiousness</b>""", unsafe_allow_html=True)
        st.write("Individu yang memiliki Dimensi Kepribadian Conscientiousness ini cenderung lebih berhati-hati dalam melakukan suatu tindakan ataupun penuh pertimbangan dalam mengambil sebuah keputusan, mereka juga memiliki disiplin diri yang tinggi dan dapat dipercaya. Karakteristik Positif pada dimensi  adalah dapat diandalkan, bertanggung jawab, tekun dan berorientasi pada pencapain.")
        st.write("Sifat kebalikan dari Conscientiousness adalah individu yang cendurung kurang bertanggung jawab, terburu-buru, tidak teratur dan kurang dapat diandalkan dalam melakukan suatu pekerjaan.")
        st.markdown(
            """<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    elif b == 'Extraversion':
        st.markdown(
            """<p style="font-size:medium;color:#e74c3c;"/p><b>Extraversion</b>""", unsafe_allow_html=True)
        st.write("Dimensi Kepribadian Extraversion ini berkaitan dengan tingkat kenyamanan seseorang dalam berinteraksi dengan orang lain. Karakteristik Positif Individu Extraversion adalah  senang bergaul, mudah bersosialisasi, hidup berkelompok dan tegas.")
        st.write("Sebaliknya, Individu yang Introversion (Kebalikan dari Extraversion) adalah mereka yang pemalu, suka menyendiri, penakut dan pendiam.")
        st.markdown(
            """<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    else:
        st.markdown(
            """<p style="font-size:medium;color:#e74c3c;"/p><b>Openness</b>""", unsafe_allow_html=True)
        st.write("Dimensi Kepribadian Opennes ini mengelompokan individu berdasarkan ketertarikannya terhadap hal-hal baru dan keinginan untuk mengetahui serta mempelajari sesuatu yang baru. Karakteristik positif pada Individu yang memiliki dimensi ini cenderung lebih kreatif, Imajinatif, Intelektual, penasaran dan berpikiran luas.")
        st.write("Sifat kebalikan dari “Openness” ini adalah individu yang cenderung konvensional dan nyaman terhadap hal-hal yang telah ada serta akan menimbulkan kegelisahan jika diberikan tugas-tugas baru.")
        st.markdown(
            """<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    st.markdown("**Nilai Akurasi Pengujian & Pelatihan Datashet :**")
    trainX, testX, trainY, testY = train_test_split(hitung, target)
    classifier.score(hitung, target)
    Scores_ml = {}
    Scores_ml['MultinomialNB'] = np.round(classifier.score(testX, testY), 5)
    st.write('Training Accuracy :', classifier.score(trainX, trainY))
    st.write('Testing Accuracy :', classifier.score(testX, testY))
    con_mat = pd.DataFrame(confusion_matrix(classifier.predict(testX), testY),
                           columns=['Predicted:Agreeableness', 'Predicted:Conscientiousness',
                                    'Predicted:Openness', 'Predicted:Neuroticism', 'Predicted:Extraversion'],
                           index=['Actual:Agreeableness', 'Actual:Conscientiousness', 'Actual:Openness', 'Actual:Neuroticism', 'Actual:Extraversion'])
    st.write('\nCLASSIFICATION REPORT\n')
    st.text(classification_report(classifier.predict(testX), testY,
                                  target_names=['Agreeableness', 'Conscientiousness', 'Openness', 'Neuroticism', 'Extraversion']))


# Query API
if nama_pengguna != "":
    limit = 70
    tweets = tweepy.Cursor(api.user_timeline, screen_name=nama_pengguna,
                           count=70, tweet_mode="extended").items(limit)
    columns = ['Tweet']
    hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    data = []
    for tweet in tweets:
        data.append(tweet.full_text)
        df = pd.DataFrame(data, columns=columns)
    data = str(data)
    text_proccesing(data)
    st.subheader("User Lookup")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**Profil Pengguna**')
        st.image(tweet.user.profile_image_url, width=70)
    with col2:
        st.write("Created :", tweet.user.created_at)
        st.write("Username : @", tweet.user.screen_name)
        st.markdown("**_Lima_ Tweet Terbaru :**")
    tabel_tweet = st.table(df[:5])
    run_klasifikasi()

else:
    st.write("Masukkan username Terlebih daulu")
