# utilities.py
# βοηθητικές μέθοδοι

import tensorflow as tf
import numpy as np
import pandas as pd
# Για αφαίρεση λέξεων χωρίς αξία (stopwords)
from nltk.corpus import stopwords
stop_words = stopwords.words('greek') #nltk stop words

# Πακέτα οπτικοποίησης
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import seaborn as sns
from sklearn import preprocessing
# Προεπεξεργασία δεδομένων
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Μοντέλα μηχανικής μάθησης
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm

# Αναφορές - μετρικές
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Μέθοδος διατήρησης positive και negative (αποκοπή neutral)
def delete_neutral(dataset):
    list_text = list(dataset['Text'])
    list_sentiment = list(dataset['Sentiment'])
    
    nea_lista_text = []
    nea_lista_sentiment = []
    
    for sent, tweet in zip(list_sentiment, list_text):
        
        if sent == 'Neutral':
            nea_lista_text.append(None)
            nea_lista_sentiment.append(None)
        else:
            nea_lista_text.append(tweet)
            nea_lista_sentiment.append(sent)
        
    d = {'Text':nea_lista_text,'Sentiment':nea_lista_sentiment}
    df = pd.DataFrame(d)
    df = df.dropna()
    return df
    
    
# Μέθοδος για stopwords και lemmatize
def stopwords_lemmatize(dataset):
    import simplemma
   
    list_text = list(dataset['Text'])
    lista_without_stopwords = []
    
    for text in list_text:
        lista_without_stopwords.append(' '.join([word for word in text.split() if word not in stop_words]))
    
    dataset['Text'] = lista_without_stopwords
    
    list_text = list(dataset['Text'])
    lista_lemmatize = []
    
    for text in list_text:
        lista_lemmatize.append(' '.join([simplemma.lemmatize(word, lang='el') for word in text.split()]))
        
    dataset['Text'] = lista_lemmatize
    return dataset


# Μέθοδος tfidf_labelEncoder
# Μέθοδος tfidf_labelEncoder
def tfidf_labelEncoder(X, y, random_split, ans_test_size=0):
    print(X, y)
    if not random_split:
        X_train, X_test = X
        y_train, y_test = y 
        print(X_train, X_test)
        tfidf_vectorizer = TfidfVectorizer(max_features=500000)
        tfidf_vectorizer.fit(X_train)
        X_train = tfidf_vectorizer.transform(X_train)
        X_test = tfidf_vectorizer.transform(X_test)

        LE = LabelEncoder()
        y_train = LE.fit_transform(y_train)
        y_test = LE.fit_transform(y_test)

        print('Σχήμα δειγμάτων εκπαίδευσης: ', X_train.shape)
        print('Σχήμα ετικετών εκπαίδευσης: ', y_train.shape)
        print('Σχήμα δειγμάτων ελέγχου: ', X_test.shape)
        print('Σχήμα ετικετών ελέγχου: ', y_test.shape)
        print('\n')
    else:
        tfidf_vectorizer = TfidfVectorizer(max_features=500000)
        X = tfidf_vectorizer.fit_transform(X)

        LE = LabelEncoder()
        y = LE.fit_transform(y)

        if ans_test_size == 0.2:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ans_test_size, stratify=y, random_state=12)
        elif ans_test_size == 0.3:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ans_test_size, stratify=y, random_state=15)
        else:
            return ValueError

        print('Σχήμα δειγμάτων εκπαίδευσης: ', X_train.shape)
        print('Σχήμα ετικετών εκπαίδευσης: ', y_train.shape)
        print('Σχήμα δειγμάτων ελέγχου: ', X_test.shape)
        print('Σχήμα ετικετών ελέγχου: ', y_test.shape)
        print('\n')
    
    return X_train, X_test, y_train, y_test


# Μέθοδος word2vec_labelEncoder
def word2vec_labelEncoder(X, y, random_split, ans_dataset, ans_test_size=0):
    if not random_split:
        X_train, X_test = X

        y_train, y_test = y 
        LE = LabelEncoder()
        y_train = LE.fit_transform(y_train)
        y_test = LE.fit_transform(y_test)
        
        print('Σχήμα δειγμάτων εκπαίδευσης: ', X_train.shape)
        print('Σχήμα ετικετών εκπαίδευσης: ', y_train.shape)
        print('Σχήμα δειγμάτων ελέγχου: ', X_test.shape)
        print('Σχήμα ετικετών ελέγχου: ', y_test.shape)
        print('\n')
        
        import gensim
        model = gensim.models.KeyedVectors.load_word2vec_format('cc.el.300.vec', limit=1000000) #limit=2000000 default
        
        # Μήκος λέξης κάθε tweet. Σε αυτό το μήκος διατηρείται κάθε δείγμα tweet.
        if ans_dataset == 'politics':
            MAX_LEN = 80   #70
        else:
            MAX_LEN = 150
        # Πλήθος αριθμών μέσα στο Vector κάθε λέξης. Κάθε λέξη έχει ένα αντίστοιχο Vector που περιέχει αριθμούς.
        VEC_LEN = 300
        
        Xtrain = tokenize(X_train, MAX_LEN, model)
        print('\n')
        Xtest = tokenize(X_test, MAX_LEN, model)
    
        print('Σχήμα δειγμάτων εκπαίδευσης 3D: ', Xtrain.shape)
        
        # Μετατροπή πίνακα 3D σε 2D 
        Xtrain = np.reshape(Xtrain,(Xtrain.shape[0],MAX_LEN*VEC_LEN))
        
        print('Σχήμα δειγμάτων εκπαίδευσης 2D: ', Xtrain.shape)
        
        print('Σχήμα δειγμάτων ελέγχου 3D: ', Xtest.shape)
        
        # Μετατροπή πίνακα 3D σε 2D 
        Xtest = np.reshape(Xtest,(Xtest.shape[0],MAX_LEN*VEC_LEN))
        
        print('Σχήμα δειγμάτων ελέγχου 2D: ', Xtest.shape)
    else:
        import gensim
        LE = LabelEncoder()
        y = LE.fit_transform(y)

        
        if ans_test_size == 0.2:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ans_test_size, stratify=y, random_state=12)
        elif ans_test_size == 0.3:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ans_test_size, stratify=y, random_state=15)
        else:
            return ValueError

        print('Σχήμα δειγμάτων εκπαίδευσης: ', X_train.shape)
        print('Σχήμα ετικετών εκπαίδευσης: ', y_train.shape)
        print('Σχήμα δειγμάτων ελέγχου: ', X_test.shape)
        print('Σχήμα ετικετών ελέγχου: ', y_test.shape)
        print('\n')


        model = gensim.models.KeyedVectors.load_word2vec_format('cc.el.300.vec', limit=100000) #limit=2000000 default

        # Μήκος λέξης κάθε tweet. Σε αυτό το μήκος διατηρείται κάθε δείγμα tweet.
        if ans_dataset == 'politics':
            MAX_LEN = 80   #70
        else:
            MAX_LEN = 150
        # Πλήθος αριθμών μέσα στο Vector κάθε λέξης. Κάθε λέξη έχει ένα αντίστοιχο Vector που περιέχει αριθμούς.
        VEC_LEN = 300

        Xtrain = tokenize(X_train, MAX_LEN, model)
        print('\n')
        Xtest = tokenize(X_test, MAX_LEN, model)

        print('Σχήμα δειγμάτων εκπαίδευσης 3D: ', Xtrain.shape)

        # Μετατροπή πίνακα 3D σε 2D 
        Xtrain = np.reshape(Xtrain,(Xtrain.shape[0],MAX_LEN*VEC_LEN))

        print('Σχήμα δειγμάτων εκπαίδευσης 2D: ', Xtrain.shape)

        print('Σχήμα δειγμάτων ελέγχου 3D: ', Xtest.shape)

        # Μετατροπή πίνακα 3D σε 2D 
        Xtest = np.reshape(Xtest,(Xtest.shape[0],MAX_LEN*VEC_LEN))

        print('Σχήμα δειγμάτων ελέγχου 2D: ', Xtest.shape)
    return Xtrain, Xtest, y_train, y_test
    
    
# Μέθοδος για τον πίνακα αντιστοιχίσεων (αριθμών) απο τις ενσωματώσεις λέξεων.
def tokenize(samples, max_len, model):
    list_token = []
    text = list(samples)
    
    for tweet in text:
        list_word = []
        for word in tweet.split():
            
            if word in model:
                my_vec = model.get_vector(word)
                #my_vec = my_vec * 100000000
                list_word.append(my_vec)
           
        list_token.append(list_word)
    
    pin = tf.keras.preprocessing.sequence.pad_sequences(list_token, maxlen=max_len, dtype='float32')
    print(pin)
    return pin   


# Μέθοδος επιλογής μοντέλου
def select_model(X_train, X_test, y_train, y_test, ans_class, ans_models, ans_embeddings):
    
    if ans_models == 'rf':
        label = 'RandomForest'
        model = RandomForestClassifier(n_estimators = 200)
        prediction, real = processing(model, X_train, X_test, y_train, y_test)
    elif ans_models == 'dt':
        label = 'DecisionTree'
        model = DecisionTreeClassifier(max_depth = 150)
        prediction, real = processing(model, X_train, X_test, y_train, y_test)
    elif ans_models == 'kn':
        label = 'KNeighbors'
        if ans_embeddings == 'tfidf':
            model = KNeighborsClassifier(n_neighbors = 3) # για tfidf 3 γείτονες
        else:
            model = KNeighborsClassifier(n_neighbors = 2, weights='distance') # για word2vec 2 γείτονες
        prediction, real = processing(model, X_train, X_test, y_train, y_test)
    elif ans_models == 'mnb':
        label = 'MultinomialNB'
        model = MultinomialNB()
        XtrainX = X_train - np.min(X_train)
        XtestX = X_test - np.min(X_test)
        prediction, real = processing(model, XtrainX, XtestX, y_train, y_test)
    elif ans_models == 'lr':
        label = 'LogisticRegression'
        model = LogisticRegression(random_state = 5, solver='lbfgs', max_iter=1000)
        prediction, real = processing(model, X_train, X_test, y_train, y_test)
    elif ans_models == 'svm':
        label = 'SupportVectorMachine'
        model = svm.SVC()
        prediction, real = processing(model, X_train, X_test, y_train, y_test)
    elif ans_models == 'gnb': 
        label = 'GaussianNB'
        model = GaussianNB()
        
        if ans_embeddings == 'tfidf':
            trainX = X_train.toarray()
            model.fit(trainX, y_train)
            testX = X_test.toarray()
            prediction = model.predict(testX)
            print('15 ετικέτες πρόβλεψης:     ', prediction[:15])
            real = y_test
            print('15 πραγματικές ετικέτες: ', real[:15], '\n')
        else: # αν ans_embeddings == word2vec
            prediction, real = processing(model, X_train, X_test, y_train, y_test)
            
    m,r = report(real, prediction, ans_class) 
    awx_confusion_matrix(real, prediction, label, ans_class)    
    return r,m
        
# Μέθοδος processing         
def processing(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print('15 ετικέτες πρόβλεψης:     ', prediction[:15])
    real = y_test
    print('15 πραγματικές ετικέτες: ', real[:15], '\n')
    return prediction, real    


# Μέθοδος report για classification report
def report(real, prediction, ans_class, transformers=False):   
    if ans_class == 'binary':
        tn = ['Negative', 'Positive']
    else:
        tn = ['Negative', 'Neutral', 'Positive']
    if transformers:
        cm = confusion_matrix(real.argmax(axis=1), prediction.argmax(axis=1))
    else:
        cm = confusion_matrix(real, prediction)
    cr = str(classification_report(real, prediction))
    cm = np.array2string(cm)
    return cm, cr
    

# Μέθοδος awx_confusion_matrix για προβολή χάρτη θερμότητας      
def awx_confusion_matrix(real, prediction, label, ans_class):
    
    fig, ax = plt.subplots(figsize=(8,8))
    
    if ans_class == 'binary':
        tn = ['Negative', 'Positive']
    else:
        tn = ['Negative', 'Neutral', 'Positive']
        
    labels = tn
    ax = sns.heatmap(confusion_matrix(real, prediction), annot=True, cmap="Reds", fmt='g', cbar=True, annot_kws={"size":17})
    plt.title(label, fontsize=20)
    ax.xaxis.set_ticklabels(labels, fontsize=17) 
    ax.yaxis.set_ticklabels(labels, fontsize=17)
    ax.set_ylabel('Real', fontsize=16)
    ax.set_xlabel('Predicted', fontsize=16)
    ax.figure.savefig('plot.png')
    return
