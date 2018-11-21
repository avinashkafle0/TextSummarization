import os

import re
import pickle
import nltk
import numpy as np
import pandas as pd
import datetime
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from PyQt4.QtGui import QFileDialog


stoplist = stopwords.words('english')
import nltk.data
from nltk.corpus import brown
from nltk import FreqDist
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Noun Part of Speech Tags used by NLTK
# More can be found here
# http://www.winwaed.com/blog/2011/11/08/part-of-speech-tags/
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']

caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1194, 651)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8("C:/Users/HP/Pictures/summarize.PNG")), QtGui.QIcon.Normal,
                       QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet(_fromUtf8("color: rgb(245, 244, 255);\n"
                                           "border-color: rgb(203, 215, 255);"))
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(60, 30, 101, 20))
        self.label.setStyleSheet(_fromUtf8("font: 11pt \"MS Shell Dlg 2\";\n"
                                           "color: rgb(0, 0, 0);"))
        self.label.setObjectName(_fromUtf8("label"))
        self.fileNameButton = QtGui.QPushButton(self.centralwidget)
        self.fileNameButton.setGeometry(QtCore.QRect(170, 30, 75, 23))
        self.fileNameButton.setStyleSheet(_fromUtf8("font: 10pt \"MS Shell Dlg 2\";\n"
                                                    "color: rgb(0, 0, 0);"))
        self.fileNameButton.setObjectName(_fromUtf8("fileNameButton"))
        self.fileNameButton.clicked.connect(self.file_Selector)
        self.textContentArea = QtGui.QTextEdit(self.centralwidget)
        self.textContentArea.setGeometry(QtCore.QRect(50, 70, 511, 491))
        self.textContentArea.setStyleSheet(_fromUtf8("font: 14pt \"Times New Roman\";\n""color: rgb(0, 0, 0);"))
        self.textContentArea.setObjectName(_fromUtf8("textContentArea"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(50, 570, 101, 16))
        self.label_2.setStyleSheet(_fromUtf8("font: 11pt \"MS Shell Dlg 2\";\n"
                                             "color: rgb(0, 0, 0);"))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.noOfSentencesArea = QtGui.QLineEdit(self.centralwidget)
        self.noOfSentencesArea.setGeometry(QtCore.QRect(150, 570, 31, 20))
        self.noOfSentencesArea.setObjectName(_fromUtf8("noOfSentencesArea"))
        self.noOfSentencesArea.setStyleSheet(_fromUtf8("font: 11pt \"MS Shell Dlg 2\";\n"
                                                       "color: rgb(0, 0, 0);"))
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(190, 570, 71, 16))
        self.label_3.setStyleSheet(_fromUtf8("font: 11pt \"MS Shell Dlg 2\";\n"
                                             "color: rgb(0, 0, 0);"))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.summarizeButton = QtGui.QPushButton(self.centralwidget)
        self.summarizeButton.setGeometry(QtCore.QRect(430, 570, 111, 41))
        self.summarizeButton.setStyleSheet(_fromUtf8("font: 14pt \"MS Shell Dlg 2\";\n"
                                                     "font: 14pt \"MS Shell Dlg 2\";\n"
                                                     "color: rgb(0, 0, 0);\n"
                                                     "background-color: rgb(202, 202, 255);"))
        self.summarizeButton.setObjectName(_fromUtf8("summarizeButton"))
        self.summarizeButton.clicked.connect(self.summarize)
        self.summarizedTextArea = QtGui.QTextEdit(self.centralwidget)
        self.summarizedTextArea.setGeometry(QtCore.QRect(610, 90, 531, 471))
        self.summarizedTextArea.setStyleSheet(_fromUtf8("font: 14pt \"Times New Roman\";\n""color: rgb(0, 0, 0);"))
        self.summarizedTextArea.setObjectName(_fromUtf8("summarizedTextArea"))
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(610, 70, 71, 16))
        self.label_4.setStyleSheet(_fromUtf8("font: 11pt \"MS Shell Dlg 2\";\n"
                                             "color: rgb(0, 0, 0);"))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1194, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Text Summarizer", None))
        self.label.setText(_translate("MainWindow", "Choose the file", None))
        self.fileNameButton.setText(_translate("MainWindow", "File name", None))
        self.label_2.setText(_translate("MainWindow", "Summarize in", None))
        self.label_3.setText(_translate("MainWindow", "sentences", None))
        self.summarizeButton.setText(_translate("MainWindow", "Summarize", None))
        self.label_4.setText(_translate("MainWindow", "Summary", None))

    def file_Selector(self):
        with open("C:/nltk_data/corpora/brown1.txt", 'r') as fileobj:
            data = fileobj.read()

            # Load the document you wish to summarize
        self.title = 'business'
        # with open("C:/nltk_data/corpora/sentences.txt", 'r') as file:
        filePath = QFileDialog.getOpenFileName()
        with open(filePath, 'r') as file:
            document = file.read()
            self.textContentArea.setText(document)
        #print(len(nltk.sent_tokenize(document)))
        self.cleaned_document = self.clean_document(document)
        self.doc = self.remove_stop_words(self.cleaned_document)
        # print(doc)

        # Merge corpus data and new document data
        filenames = ["C:/nltk_data/corpora/brown1.txt", filePath]
        with open("C:/nltk_data/corpora/document1.txt", 'w') as outfile:  # here document3 is file name
            for fname in filenames:
                with open(fname) as infile:
                    outfile.write(infile.read())

        train_data = open("C:/nltk_data/corpora/document1.txt", 'r')

        # Fit and Transform the term frequencies into a vector
        count_vect = CountVectorizer()
        count_vect = count_vect.fit(train_data)
        # print(count_vect)
        #print(count_vect.get_feature_names())

        with open("C:/nltk_data/corpora/document3.txt") as f:
            train_data = list(f)

        train_data = self.split_into_sentences(train_data)

        freq_term_matrix = count_vect.transform(train_data)
        print(freq_term_matrix.toarray)

        self.feature_names = count_vect.get_feature_names()

        # Fit and Transform the TfidfTransformer
        tfidf = TfidfTransformer(norm="l2")
        tfidf.fit(freq_term_matrix)
        #print(t.toshape())
        # Get the dense tf-idf matrix for the document
        story_freq_term_matrix = count_vect.transform([self.doc])
        print(story_freq_term_matrix)
        story_tfidf_matrix = tfidf.transform(story_freq_term_matrix)
        print(story_tfidf_matrix.shape)
        story_dense = story_tfidf_matrix.todense()
        print(story_dense)
        self.doc_matrix = story_dense.tolist()[0]
        # print(doc_matrix)
        # Get Top Ranking Sentences and join them as a summary
        #top_sents = self.rank_sentences(doc, doc_matrix, feature_names)
        #summary = '.'.join([cleaned_document.split('.')[i] for i in [pair[0] for pair in top_sents]])
        #summary = ' '.join(summary.split())



    def summarize(self):
        top_sents = self.rank_sentences(self.doc, self.doc_matrix, self.feature_names)
        summary = '.'.join([self.cleaned_document.split('.')[i] for i in [pair[0] for pair in top_sents]])
        summary = ' '.join(summary.split())
        self.summarizedTextArea.setText(summary)

    def split_into_sentences(self,text):
        text = " " + str(text) + "  "
        text = text.replace("\n", " ")
        text = re.sub(prefixes, "\\1<prd>", text)
        text = re.sub(websites, "<prd>\\1", text)
        if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
        text = re.sub("\s" + caps + "[.] ", " \\1<prd> ", text)
        text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
        text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
        text = re.sub(caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>", text)
        text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
        text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
        text = re.sub(" " + caps + "[.]", " \\1<prd>", text)
        if "”" in text: text = text.replace(".”", "”.")
        if "\"" in text: text = text.replace(".\"", "\".")
        if "!" in text: text = text.replace("!\"", "\"!")
        if "?" in text: text = text.replace("?\"", "\"?")
        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")
        text = text.replace("<prd>", ".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        return sentences

    def clean_document(self,document):
        """Cleans document by removing unnecessary punctuation. It also removes
        any extra periods and merges acronyms to prevent the tokenizer from
        splitting a false sentence
        """
        # Remove all characters outside of Alpha Numeric
        # and some punctuation
        document = re.sub('[^A-Za-z .-]+', ' ', document)
        document = document.replace('-', '')
        document = document.replace('...', '')
        document = document.replace('Mr.', 'Mr').replace('Mrs.', 'Mrs')

        # Remove Ancronymns M.I.T. -> MIT
        # to help with sentence tokenizing
        document = self.merge_acronyms(document)

        # Remove extra whitespace
        document = ' '.join(document.split())
        return document

    def remove_stop_words(self,document):
        # """Returns document without stop words"""
        document = ' '.join([i for i in document.split() if i not in stoplist])
        return document

    def similarity_score(self,t, s):
        """Returns a similarity score for a given sentence.
        similarity score = the total number of tokens in a sentence that exits
                            within the title / total words in title
        """
        t = self.remove_stop_words(t.lower())
        s = self.remove_stop_words(s.lower())
        t_tokens, s_tokens = t.split(), s.split()
        similar = [w for w in s_tokens if w in t_tokens]
        score = (len(similar) * 0.1) / len(t_tokens)
        return score

    def merge_acronyms(self,s):
        """Merges all acronyms in a given sentence. For example M.I.T -> MIT"""
        r = re.compile(r'(?:(?<=\.|\s)[A-Z]\.)+')
        acronyms = r.findall(s)
        for a in acronyms:
            s = s.replace(a, a.replace('.', ''))
        return s

    def rank_sentences(self,doc, doc_matrix, feature_names, top_n=1):
        """
        Returns top_n sentences. Theses sentences are then used as summary
        of document.
        input
        ------------
        doc : a document as type str
        doc_matrix : a dense tf-idf matrix calculated with Scikits TfidfTransformer
        feature_names : a list of all features, the index is used to look up
                        tf-idf scores in the doc_matrix
        top_n : number of sentences to return
        """
        sents = nltk.sent_tokenize(doc)
        sentences = [nltk.word_tokenize(sent) for sent in sents]
        sentences = [[w for w in sent if nltk.pos_tag([w])[0][1] in NOUNS]
                     for sent in sentences]
        tfidf_sent = [[doc_matrix[feature_names.index(w.lower())]
                       for w in sent if w.lower() in feature_names] for sent in sentences]

        # Calculate Sentence Values
        doc_val = sum(doc_matrix)
        sent_values = [(sum(sent) / doc_val) for sent in tfidf_sent]
        #print(sent_values)
        # Apply Similariy Score Weightings
        similarity_scores = [self.similarity_score(self.title, sent) for sent in sents]
        scored_sents = np.array(sent_values) + np.array(similarity_scores)
        # print(scored_sents)
        # Apply Position Weights
        ranked_sents = [sent * (i / len(sent_values)) for i, sent in enumerate(sent_values)]

        ranked_sents = [pair for pair in zip(range(len(sent_values)), sent_values)]
        no = int(self.noOfSentencesArea.text())
        ranked_sents = sorted(ranked_sents, key=lambda x: x[1] * -1)
        return ranked_sents[:no]


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

