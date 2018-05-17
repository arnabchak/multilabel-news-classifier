import tkinter as Tk


def classification(X_in):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.svm import LinearSVC
        from sklearn.feature_extraction.text import TfidfTransformer
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import MultiLabelBinarizer

        dataset = pd.read_csv('Book1.csv')

        import re
        import nltk
        from nltk.corpus import stopwords
        corpus = []
        for i in range(0, 72):
            headline = re.sub('[^a-zA-Z]', ' ', dataset['headlines'][i])
            headline = headline.lower()
            headline = headline.split()

            headline = [word for word in headline if not word
                        in set(stopwords.words('english'))]
            headline = ' '.join(headline)
            corpus.append(headline)

        X = np.array(corpus)
        y = list(dataset.category.str.split('|'))

          # yvalues = np.array(y)

        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(y)

        from sklearn.cross_validation import train_test_split
        (X_train, X_test, y_train, y_test) = train_test_split(X, Y,
                test_size=0.3)

        classifier1 = Pipeline([('vectorizer', CountVectorizer()),
                               ('tfidf', TfidfTransformer()), ('clf',
                               OneVsRestClassifier(LinearSVC()))])

        classifier1.fit(X_train, y_train)
        predicted = classifier1.predict(X_test)

        from sklearn.metrics import accuracy_score

        print('the accuracy is:')
        print( accuracy_score(y_test,predicted))

        X_input = []
        X_input.append(X_in)

        classifier2 = Pipeline([('vectorizer', CountVectorizer()),
                               ('tfidf', TfidfTransformer()), ('clf',
                               OneVsRestClassifier(LinearSVC()))])

        classifier2.fit(X, Y)
        predicted_input = classifier2.predict(X_input)
        all_labels = mlb.inverse_transform(predicted_input)
        return all_labels
        #for (item, labels) in zip(X_input, all_labels):
         #   print ('{1}'.format(item, ', '.join(labels)))

class App(object):

    def __init__(self):
        self.root = Tk.Tk()
        self.root.wm_title('News Classifier')
        self.label = Tk.Label(self.root, text='Enter the headline.')
        self.label.pack()
        self.string = Tk.StringVar()
        Tk.Entry(self.root, textvariable=self.string).pack()
        self.buttontext = Tk.StringVar()
        self.buttontext.set('Enter')

        Tk.Button(self.root, textvariable=self.buttontext,
                  command=self.clicked1).pack()
        self.label = Tk.Label(self.root, text='')
        self.label.pack()
        self.root.mainloop()

    def clicked1(self):
        headline = self.string.get()
        category=classification(headline)
        for c in category:
            self.label.configure(text=c)

    def button_click(self, e):
        pass

App()
