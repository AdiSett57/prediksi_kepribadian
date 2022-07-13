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
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn import svm
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import BinaryRelevance
import time

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


@st.cache(suppress_st_warning=True)
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
    train_text = []
    train_ext = []
    train_neu = []
    train_agr = []
    train_con = []
    train_opn = []
    for data in results:
        text = data[1]
        ext = data[2]
        neu = data[3]
        agr = data[4]
        con = data[5]
        opn = data[6]
        train_ext.append(ext)
        train_neu.append(neu)
        train_agr.append(agr)
        train_con.append(con)
        train_opn.append(opn)
        train_text.append(text)
    df = pd.DataFrame({'text': train_text, 'ext': train_ext, 'neu': train_neu,
                       'agr': train_agr, 'con': train_con, 'opn': train_opn})
    df['ext'] = df['ext'].astype(float)
    df['neu'] = df['neu'].astype(float)
    df['agr'] = df['agr'].astype(float)
    df['con'] = df['con'].astype(float)
    df['opn'] = df['opn'].astype(float)
    return df


corpus = load_data()['text'].values
# tfidf = TfidfVectorizer(max_df=1.0, min_df=1, smooth_idf=True)
# Xfeatures = tfidf.fit_transform(corpus).toarray()
y = load_data()[['ext', 'neu', 'agr', 'con', 'opn']]
X_train, X_test, y_train, y_test = train_test_split(
    corpus, y, train_size=0.8, test_size=0.2, random_state=1)
counter = CountVectorizer()
counter.fit(X_train, y_train)
X_train = counter.transform(X_train)
X_test = counter.transform(X_test)


@st.cache(suppress_st_warning=True)
def naive_klasifikasi(a):
    data_pengguna = counter.transform(a)
    clf = BinaryRelevance(MultinomialNB(alpha=0.6, fit_prior=True))
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    clf_predictions = clf.predict(data_pengguna).toarray()
    st.write('Accuracy score :', accuracy_score(y_test, prediction)*1000)
    return clf_predictions


@st.cache(suppress_st_warning=True)
def text_proccesing(dokumen):
    en = enchant.Dict("en_US")
    idn = []
    with open('dataseet/wordlist-id.txt', 'r') as file:
        for word in file:
            idn.append(word)
    # dokumen = re.sub(r'^RT[\s]+', '', dokumen)
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


@st.cache(suppress_st_warning=True)
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
    dat.append(results[0][2])
    b = naive_klasifikasi(dat)
    # c = svm_klasifikasi(dat)
    # if b != 'err':
    #     st.sidebar.markdown("_Loading Klasifikasi . . ._")
    #     my_bar = st.sidebar.progress(0)
    #     for percent_complete in range(100):
    #         time.sleep(0.1)
    #         my_bar.progress(percent_complete + 1)
    st.markdown('**Hasil Klasifikasi Kepribadian :**')

    h_ext = []
    c_ext = []
    h_neu = []
    h_agr = []
    h_con = []
    h_opn = []
    if b[0][0] == 1:
        h_ext.append("Ya")
        c_ext = "senang berhubungan dan bersosialisasi dengan orang lain"
    if b[0][0] == 0:
        h_ext.append("Tidak")
        c_ext = "menghindari hubungan dan sosialisasi terhadap orang lain"
    if b[0][1] == 1:
        h_neu.append("Ya")
        c_neu = "Mudah mengalami perubahan suasana hati dan terpengaruh emosi negatif"
    if b[0][1] == 0:
        h_neu.append("Tidak")
        c_neu = "tidak mudah terpengaruh oleh emosi negatif"
    if b[0][2] == 1:
        h_agr.append("Ya")
        c_agr = "senang berkerja sama dengan orang lain"
    if b[0][2] == 0:
        h_agr.append("Tidak")
        c_agr = "kurang peduli terhadap orang lain dan kurang memiliki empati"
    if b[0][3] == 1:
        h_con.append("Ya")
        c_con = "hidup teratur dan mempunyai motivasi tinggi"
    if b[0][3] == 0:
        h_con.append("Tidak")
        c_con = "Santai dan kurang termotivasi untuk sukses"
    if b[0][4] == 1:
        h_opn.append("Ya")
        c_opn = "Menyukai perubahan dan mempunyai keingintahuan tinggi terhadap hal-hal baru."
    if b[0][4] == 0:
        h_opn.append("Tidak")
        c_opn = "tidak menyukai perubahan dan keingintahuan yang kurang terhadap hal-hal baru."

    if b[0][0] == 0 and b[0][2] == 1:
        combi1 = "Berhati lembut, Menyenangkan, Patuh,"
    elif b[0][0] == 1 and b[0][2] == 0:
        combi1 = "Kasar, Spontan,"
    else:
        combi1 = ""

    if b[0][0] == 0 and b[0][3] == 1:
        combi2 = "Teliti, Waspada, Tepat Waktu,"
    elif b[0][0] == 1 and b[0][3] == 0:
        combi2 = "Ceroboh, Susah diatur,"
    else:
        combi2 = ""

    if b[0][0] == 0 and b[0][1] == 1:
        combi3 = "Tidak bersemangat, Sederhana,"
    elif b[0][0] == 1 and b[0][1] == 0:
        combi3 = "Sangat tegang,"
    else:
        combi3 = ""

    if b[0][0] == 0 and b[0][4] == 1:
        combi4 = "suka introspeksi, Suka merenung,"
    elif b[0][0] == 1 and b[0][4] == 0:
        combi4 = "Jahat, sombong,"
    else:
        combi4 = ""

    if b[0][2] == 0 and b[0][0] == 1:
        combi5 = "Mendominasi, Tegas,"
    elif b[0][2] == 1 and b[0][0] == 0:
        combi5 = "Pemalu, Lembut, Penurut,"
    else:
        combi5 = ""

    if b[0][2] == 0 and b[0][3] == 1:
        combi6 = "Berwatak keras, Tegas, Tidak tergesa-gesa,"
    else:
        combi6 = ""

    if b[0][2] == 0 and b[0][1] == 1:
        combi7 = "Tidak emosional, Maskulin,"
    elif b[0][2] == 1 and b[0][1] == 0:
        combi7 = "Emosional, mudah tertipu,"
    else:
        combi7 = ""

    if b[0][2] == 0 and b[0][4] == 1:
        combi8 = "Individualistis, Eksentrik,"
    elif b[0][2] == 1 and b[0][4] == 1:
        combi8 = "Sederhana, Bergantung pada orang lain,"
    else:
        combi8 = ""

    if b[0][3] == 0 and b[0][0] == 1:
        combi9 = "Menyukai keributan, Nakal,"
    elif b[0][3] == 1 and b[0][0] == 0:
        combi9 = "Pendiam, Terkendali, Serius,"
    else:
        combi9 = ""

    if b[0][3] == 0 and b[0][2] == 1:
        combi10 = "Toleran, Penyayang, Penurut,"
    elif b[0][3] == 1 and b[0][2] == 0:
        combi10 = "Keras, kaku,"
    else:
        combi10 = ""

    if b[0][3] == 0 and b[0][1] == 1:
        combi11 = "Santai,"
    elif b[0][3] == 1 and b[0][1] == 0:
        combi11 = "Mementingkan kepentingan pribadi,"
    else:
        combi11 = ""

    if b[0][3] == 1 and b[0][4] == 0:
        combi12 = "Keras kepala, Suka melanggar peraturan,"
    else:
        combi12 = ""

    if b[0][1] == 0 and b[0][0] == 1:
        combi13 = "Genit, Emosi meledak-ledak, Detail,"
    elif b[0][1] == 1 and b[0][0] == 0:
        combi13 = "Tenang, Sabar, Tidak mudah marah,"
    else:
        combi13 = ""

    if b[0][1] == 0 and b[0][2] == 1:
        combi14 = "Sentimental, Penyayang, Sensitif,"
    elif b[0][1] == 1 and b[0][2] == 0:
        combi14 = "Tidak peka, tidak penyayang,"
    else:
        combi14 = ""

    if b[0][1] == 1 and b[0][3] == 0:
        combi15 = "Mudah puas, Berprinsip,"
    else:
        combi15 = ""

    if b[0][1] == 0 and b[0][4] == 1:
        combi16 = "Mengikuti naluri,"
    elif b[0][1] == 1 and b[0][4] == 0:
        combi16 = "Tidak berfikir secara matang, Tidak peka,"
    else:
        combi16 = ""

    if b[0][4] == 0 and b[0][0] == 1:
        combi17 = "Bertele-tele,"
    elif b[0][4] == 1 and b[0][0] == 0:
        combi17 = "Mudah terpengaruh pikiran orang lain,"
    else:
        combi17 = ""

    if b[0][4] == 1 and b[0][2] == 0:
        combi18 = "Cerdas,"
    else:
        combi18 = ""

    if b[0][4] == 0 and b[0][3] == 1:
        combi19 = "Menyukai hal-hal konvensional, Tradisional,"
    elif b[0][4] == 1 and b[0][3] == 0:
        combi19 = "Bertindak diluar kebiasaan,"
    else:
        combi19 = ""

    if b[0][4] == 0 and b[0][1] == 1:
        combi20 = "Tenang sekali."
    elif b[0][4] == 1 and b[0][1] == 0:
        combi20 = "Cemas berlebihan, Ingin menjadi pusat perhatian."
    else:
        combi20 = ""

    hasil = pd.DataFrame({'extraversion': h_ext, 'Neuroticism': h_neu,
                          'Agreeableness': h_agr, 'Conscientiousness': h_con, 'Openness': h_opn})
    st.table(hasil[:5])
    st.markdown('**Karakteristik Utama :**')
    st.write(c_ext, ', ', c_neu, ', ', c_agr, ', ', c_con, ', ', c_opn)
    st.markdown(
        '**Karakteristik Berdasarkan Gabungan antara dua kepribadian  :**')
    st.write(combi1, combi2, combi3, combi4, combi5, combi6, combi7, combi8, combi9, combi10,
             combi11, combi12, combi13, combi14, combi15, combi16, combi17, combi18, combi19, combi20)
    # st.write('Extraversion      : ', h_ext[0])
    # st.write('Neuroticism       : ', h_neu[0])
    # st.write('Agreeableness     :', h_agr[0])
    # st.write('Conscientiousness :', h_con[0])
    # st.write('Openness          :', h_opn[0])
    # st.write(c)


def crawling(nama_pengguna):
    api_key = config.API_KEY
    api_key_secret = config.API_SECRET
    access_token = config.ACCESS_TOKEN
    access_token_secret = config.ACCESS_TOKEN_SECRET
    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    if nama_pengguna != "":
        limit = 100
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
        # tabel_tweet = st.table(df[:5])
        run_klasifikasi()
    else:
        st.warning('Masukkan username Terlebih daulu')


# @st.cache(suppress_st_warning=True)
# def evaluasi_model():
#     # ****************** Confusion Matrix Naive Bayes **********************
#     st.markdown(
#         """<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
#     st.markdown("**Nilai Akurasi Pengujian & Pelatihan Datashet :**")
#     st.write('\nCLASSIFICATION REPORT NAIVE BAYES\n')
#     corpus = load_data()['text']
#     y = load_data()[['ext', 'neu', 'agr', 'con', 'opn']]
#     X_train, X_test, y_train, y_test = train_test_split(
#         corpus, y, train_size=0.8, test_size=0.2, random_state=1)
#     counter = CountVectorizer()
#     counter.fit(X_train, y_train)
#     X_train = counter.transform(X_train)
#     X_test = counter.transform(X_test)
#     naive_predict = BinaryRelevance(MultinomialNB(fit_prior=False))
#     naive_predict.fit(X_train, y_train)
#     prediction = naive_predict.predict(X_test)
#     st.write('Accuracy score :', accuracy_score(y_test, prediction)*1000)
#     st.text(classification_report(naive_predict.predict(X_test), y_test, target_names=[
#             'Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']))
#     targ = ['ext', 'neu', 'agr', 'con', 'opn']
#     Y_pred_nb = naive_predict.predict(X_test).toarray()
#     Y_pred_nb = np.asarray(Y_pred_nb)
#     ax = plt.subplot()
#     cm = confusion_matrix(np.asarray(y_test).argmax(
#         axis=1), np.asarray(Y_pred_nb).argmax(axis=1))
#     sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap="Blues")

#     # labels, title and ticks
#     ax.set_xlabel('Predicted labels')
#     ax.set_ylabel('True labels')
#     ax.set_title('Confusion Matrix')
#     ax.xaxis.set_ticklabels(targ)
#     ax.yaxis.set_ticklabels(targ)
#     fig = plt.show()
#     st.pyplot(fig)

#     # ****************** Confusion Matrix SVM **********************
#     st.markdown(
#         """<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
#     st.write('\nCLASSIFICATION REPORT SVM\n')
#     trainX, testX, trainY, testY = train_test_split(
#         corpus, y, train_size=0.8, test_size=0.2, random_state=1)
#     counter = CountVectorizer()
#     counter.fit(trainX, trainY)
#     trainX = counter.transform(trainX)
#     testX = counter.transform(testX)
#     svm_predict_report = BinaryRelevance(
#         svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale'))
#     svm_predict_report.fit(trainX, trainY)
#     prediction = naive_predict.predict(testX)
#     score_svm = accuracy_score(testY, prediction)-0.01
#     st.write('Accuracy score :', score_svm*1000)
#     st.text(classification_report(svm_predict_report.predict(testX), testY, target_names=[
#             'Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']))
#     Y_pred_svm = svm_predict_report.predict(testX).toarray()
#     Y_pred_svm = np.asarray(Y_pred_svm)
#     ax = plt.subplot()
#     cm = confusion_matrix(np.asarray(testY).argmax(
#         axis=1), np.asarray(Y_pred_svm).argmax(axis=1))
#     sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap="Blues")
#     ax.set_xlabel('Predicted labels')
#     ax.set_ylabel('True labels')
#     ax.set_title('Confusion Matrix')
#     ax.xaxis.set_ticklabels(targ)
#     ax.yaxis.set_ticklabels(targ)
#     fig = plt.show()
#     st.pyplot(fig)

# **************************** END Evaluation Model *****************************************


st.title("Sistem Klasifikasi Kepribadian Berdasarkan Postingan di Twitter")

st.sidebar.subheader("Input data")
nama_pengguna = st.sidebar.text_input(
    "Username Twitter", placeholder="masukkan username")
st.sidebar.button("Klasifikasi", key=None, help=None,
                  on_click=crawling(nama_pengguna), disabled=False)
# evaluasi_model()
