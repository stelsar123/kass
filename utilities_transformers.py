# utilities_teansformers.py
# βοηθητικές μέθοδοι

import tensorflow as tf
import numpy as np
import pandas as pd
import transformers
from transformers import BertTokenizer, TFBertModel, AutoTokenizer, TFXLMRobertaModel, TFAutoModel, DistilBertTokenizer, TFDistilBertModel
from transformers import TFAutoModelForCausalLM, GPT2Tokenizer
#from tensorflow.keras.models import load_model, Model
from sklearn import preprocessing
import torch
from utilities import report, delete_neutral, stopwords_lemmatize, awx_confusion_matrix
# Για αφαίρεση λέξεων χωρίς αξία (stopwords)
from nltk.corpus import stopwords
stop_words = stopwords.words('greek') #nltk stop words



# Μέθοδος επιλογής μοντέλου
def select_model_transformers(ans_dataset, ans_class, ans_stopword_lemma, ans_models):
    
    if ans_dataset == 'politics':
        MAX_LENGTH = 80
    else:
        MAX_LENGTH = 150
    
    
    # BERT MODEL
    if ans_models == 'bert':
        label = 'Bert Model'
        headers=['Text','Sentiment']
        
        if ans_dataset == 'politics':
            dataset = pd.read_csv('Politics/train2682_test810/test_set_politics810.csv', sep=',', names=headers)
        else:
            dataset = pd.read_csv('Skroutz/test_set_skroutz1966.csv', sep=',', names=headers)
    
        if ans_dataset == 'politics' and ans_class == 'binary':
            dataset = delete_neutral(dataset)
        
        if ans_stopword_lemma == 'yes':
            dataset = stopwords_lemmatize(dataset)
        
        if ans_class == 'binary':
            dataset.Sentiment.replace("Positive", 1, inplace = True)
            dataset.Sentiment.replace("Negative", 0, inplace = True)
            y_test = dataset['Sentiment']
            X_test = dataset['Text']
        else:
            ohe = preprocessing.OneHotEncoder()
            y_test = ohe.fit_transform(np.array(dataset['Sentiment']).reshape(-1, 1)).toarray()
            X_test = dataset['Text']
        
        tokenizer = BertTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
        X_test_token = get_tokens(X_test, tokenizer, MAX_LENGTH)
        
        # Ορίζουμε το μοντέλο BERT ως υποκλάση της Model
        class MyBertModel(tf.keras.Model):
            def __init__(self):
                super(MyBertModel, self).__init__()
                self.bert = TFAutoModel.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
                self.dropout1 = tf.keras.layers.Dropout(0.2)
                self.dense1 = tf.keras.layers.Dense(64, activation='relu')
                self.dropout2 = tf.keras.layers.Dropout(0.2)
                if ans_class == 'binary':
                    self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')
                else:
                    self.dense2 = tf.keras.layers.Dense(3, activation='softmax')

            def call(self, inputs, **kwargs):
                input_ids = inputs[0]
                attention_mask = inputs[1]
                outputs = self.bert(input_ids, attention_mask, **kwargs)[1]
                outputs = self.dropout1(outputs)
                outputs = self.dense1(outputs)
                outputs = self.dropout2(outputs)
                outputs = self.dense2(outputs)
                return outputs
        
        model = MyBertModel()
        input_ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="input_ids")
        attention_mask = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="attention_mask")
        outputs = model([input_ids, attention_mask])
        
        # Φόρτωση των βαρών του προ-εκπαιδευμένου μοντέλου
        if ans_dataset == 'politics':
            if ans_class == 'binary' and ans_stopword_lemma == 'yes':
                model.load_weights('E:\\h5\\Politics\\train2682_test810\\train_bert_bin_politics_with_sl.h5')
            elif ans_class == 'binary' and ans_stopword_lemma == 'no':
                model.load_weights('E:\\h5\\Politics\\train2682_test810\\train_bert_bin_politics_without_sl.h5')
            elif ans_class == 'multi' and ans_stopword_lemma == 'yes':
                model.load_weights('E:\\h5\\Politics\\train2682_test810\\train_bert_multi_politics_with_sl.h5')
            elif ans_class == 'multi' and ans_stopword_lemma == 'no':
                model.load_weights('E:\\h5\\Politics\\train2682_test810\\train_bert_multi_politics_without_sl.h5')
        
        if ans_dataset == 'skroutz':
            if ans_stopword_lemma == 'yes':
                model.load_weights('E:\\h5\\Skroutz\\train_bert_bin_skroutz_with_sl.h5')
            else:
                model.load_weights('E:\\h5\\Skroutz\\train_bert_bin_skroutz_without_sl.h5')
        
        # Χρήση μεθόδου predict()
        if ans_class == 'binary':
            pred = np.where(model.predict([X_test_token['input_ids'], X_test_token['attention_mask']]) >= 0.5, 1, 0)
            m,r = report(y_test, pred, ans_class)
            awx_confusion_matrix(y_test, pred, label, ans_class)
        else:
            pred = model.predict([X_test_token['input_ids'], X_test_token['attention_mask']])
            y_pred_bert = np.zeros_like(pred)
            y_pred_bert[np.arange(len(y_pred_bert)), pred.argmax(1)] = 1
        
            m,r = report(y_test, y_pred_bert, ans_class, transformers=True)
            awx_confusion_matrix(y_test.argmax(1), y_pred_bert.argmax(1), label, ans_class)
            
            
    # RoBERTa MODEL
    if ans_models == 'roberta':
        label = 'RoBERTa Model'
        headers=['Text','Sentiment']
        
        if ans_dataset == 'politics':
            dataset = pd.read_csv('Politics/train2682_test810/test_set_politics810.csv', sep=',', names=headers)
        else:
            dataset = pd.read_csv('Skroutz/test_set_skroutz1966.csv', sep=',', names=headers)
    
        if ans_dataset == 'politics' and ans_class == 'binary':
            dataset = delete_neutral(dataset)
        
        if ans_stopword_lemma == 'yes':
            dataset = stopwords_lemmatize(dataset)
        
        if ans_class == 'binary':
            dataset.Sentiment.replace("Positive", 1, inplace = True)
            dataset.Sentiment.replace("Negative", 0, inplace = True)
            y_test = dataset['Sentiment']
            X_test = dataset['Text']
        else:
            ohe = preprocessing.OneHotEncoder()
            y_test = ohe.fit_transform(np.array(dataset['Sentiment']).reshape(-1, 1)).toarray()
            X_test = dataset['Text']
        
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        X_test_token = get_tokens(X_test, tokenizer, MAX_LENGTH)
        
        # Ορίζουμε το μοντέλο RoBERTa ως υποκλάση της Model
        class MyRoBERTaModel(tf.keras.Model):
            def __init__(self):
                super(MyRoBERTaModel, self).__init__()
                self.roberta = TFXLMRobertaModel.from_pretrained('jplu/tf-xlm-roberta-base')
                self.dropout1 = tf.keras.layers.Dropout(0.2)
                self.dense1 = tf.keras.layers.Dense(128, activation='relu')
                self.dropout2 = tf.keras.layers.Dropout(0.2)
                self.dense2 = tf.keras.layers.Dense(64, activation='relu')
                self.dropout3 = tf.keras.layers.Dropout(0.2)
                
                if ans_class == 'binary':
                    self.dense3 = tf.keras.layers.Dense(1, activation='sigmoid')
                else:
                    self.dense3 = tf.keras.layers.Dense(3, activation='softmax')

            def call(self, inputs, **kwargs):
                input_ids = inputs[0]
                attention_mask = inputs[1]
                outputs = self.roberta(input_ids, attention_mask, **kwargs)[1]
                outputs = self.dropout1(outputs)
                outputs = self.dense1(outputs)
                outputs = self.dropout2(outputs)
                outputs = self.dense2(outputs)
                outputs = self.dropout3(outputs)
                outputs = self.dense3(outputs)
                return outputs
        
        model = MyRoBERTaModel()
        input_ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="input_ids")
        attention_mask = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="attention_mask")
        outputs = model([input_ids, attention_mask])
        
        # Φόρτωση των βαρών του προ-εκπαιδευμένου μοντέλου
        if ans_dataset == 'politics':
            if ans_class == 'binary' and ans_stopword_lemma == 'yes':
                model.load_weights('E:\\h5\\Politics\\train2682_test810\\train_roberta_bin_politics_with_sl.h5')
            elif ans_class == 'binary' and ans_stopword_lemma == 'no':
                model.load_weights('E:\\h5\\Politics\\train2682_test810\\train_roberta_bin_politics_without_sl.h5')
            elif ans_class == 'multi' and ans_stopword_lemma == 'yes':
                model.load_weights('E:\\h5\\Politics\\train2682_test810\\train_roberta_multi_politics_with_sl.h5')
            elif ans_class == 'multi' and ans_stopword_lemma == 'no':
                model.load_weights('E:\\h5\\Politics\\train2682_test810\\train_roberta_multi_politics_without_sl.h5')
        
        if ans_dataset == 'skroutz':
            if ans_stopword_lemma == 'yes':
                model.load_weights('E:\\h5\\Skroutz\\train_roberta_bin_skroutz_with_sl.h5')
            else:
                model.load_weights('E:\\h5\\Skroutz\\train_roberta_bin_skroutz_without_sl.h5')
        
        # Χρήση μεθόδου predict()
        if ans_class == 'binary':
            pred = np.where(model.predict([X_test_token['input_ids'], X_test_token['attention_mask']]) >= 0.5, 1, 0)
            m,r = report(y_test, pred, ans_class)
            awx_confusion_matrix(y_test, pred, label, ans_class)
        else:
            pred = model.predict([X_test_token['input_ids'], X_test_token['attention_mask']])
            y_pred_roberta = np.zeros_like(pred)
            y_pred_roberta[np.arange(len(y_pred_roberta)), pred.argmax(1)] = 1
        
            m,r = report(y_test, y_pred_roberta, ans_class, True)
            awx_confusion_matrix(y_test.argmax(1), y_pred_roberta.argmax(1), label, ans_class)
        
        
    # DistilBERT MODEL
    if ans_models == 'distilbert':
        label = 'DistilBERT Model'
        headers=['Text','Sentiment']
        
        if ans_dataset == 'politics':
            dataset = pd.read_csv('Politics/train2682_test810/test_set_politics810.csv', sep=',', names=headers)
        else:
            dataset = pd.read_csv('Skroutz/test_set_skroutz1966.csv', sep=',', names=headers)
    
        if ans_dataset == 'politics' and ans_class == 'binary':
            dataset = delete_neutral(dataset)
        
        if ans_stopword_lemma == 'yes':
            dataset = stopwords_lemmatize(dataset)
        
        if ans_class == 'binary':
            dataset.Sentiment.replace("Positive", 1, inplace = True)
            dataset.Sentiment.replace("Negative", 0, inplace = True)
            y_test = dataset['Sentiment']
            X_test = dataset['Text']
        else:
            ohe = preprocessing.OneHotEncoder()
            y_test = ohe.fit_transform(np.array(dataset['Sentiment']).reshape(-1, 1)).toarray()
            X_test = dataset['Text']
        
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
        X_test_token = get_tokens(X_test, tokenizer, MAX_LENGTH)
        
        # Ορίζουμε το μοντέλο DistilBERT ως υποκλάση της Model
        class MyDistilBERTModel(tf.keras.Model):
            def __init__(self):
                super(MyDistilBERTModel, self).__init__()
                self.distilbert = TFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased', from_pt=True)
                self.dropout1 = tf.keras.layers.Dropout(0.2)
                self.dense1 = tf.keras.layers.Dense(256, activation='relu')
                self.dropout2 = tf.keras.layers.Dropout(0.2)
                self.dense2 = tf.keras.layers.Dense(128, activation='relu')
                self.dropout3 = tf.keras.layers.Dropout(0.2)
                self.dense3 = tf.keras.layers.Dense(64, activation='relu')
                self.dropout4 = tf.keras.layers.Dropout(0.2)
                self.flatten = tf.keras.layers.Flatten()
                
                if ans_class == 'binary':
                    self.dense4 = tf.keras.layers.Dense(1, activation='sigmoid')
                else:
                    self.dense4 = tf.keras.layers.Dense(3, activation='softmax')

            def call(self, inputs, **kwargs):
                input_ids = inputs[0]
                attention_mask = inputs[1]
                outputs = self.distilbert(input_ids, attention_mask, **kwargs)[0]
                outputs = self.dropout1(outputs)
                outputs = self.dense1(outputs)
                outputs = self.dropout2(outputs)
                outputs = self.dense2(outputs)
                outputs = self.dropout3(outputs)
                outputs = self.dense3(outputs)
                outputs = self.dropout4(outputs)
                outputs = self.flatten(outputs)
                outputs = self.dense4(outputs)
                return outputs
        
        model = MyDistilBERTModel()
        input_ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="input_ids")
        attention_mask = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="attention_mask")
        outputs = model([input_ids, attention_mask])
        
        # Φόρτωση των βαρών του προ-εκπαιδευμένου μοντέλου
        if ans_dataset == 'politics':
            if ans_class == 'binary' and ans_stopword_lemma == 'yes':
                model.load_weights('E:\\h5\\Politics\\train2682_test810\\train_distilbert_bin_politics_with_sl.h5')
            elif ans_class == 'binary' and ans_stopword_lemma == 'no':
                model.load_weights('E:\\h5\\Politics\\train2682_test810\\train_distilbert_bin_politics_without_sl.h5')
            elif ans_class == 'multi' and ans_stopword_lemma == 'yes':
                model.load_weights('E:\\h5\\Politics\\train2682_test810\\train_distilbert_multi_politics_with_sl.h5')
            elif ans_class == 'multi' and ans_stopword_lemma == 'no':
                model.load_weights('E:\\h5\\Politics\\train2682_test810\\train_distilbert_multi_politics_without_sl.h5')
        
        if ans_dataset == 'skroutz':
            if ans_stopword_lemma == 'yes':
                model.load_weights('E:\\h5\\Skroutz\\train_distilbert_bin_skroutz_with_sl.h5')
            else:
                model.load_weights('E:\\h5\\Skroutz\\train_distilbert_bin_skroutz_without_sl.h5')
        
        # Χρήση μεθόδου predict()
        if ans_class == 'binary':
            pred = np.where(model.predict([X_test_token['input_ids'], X_test_token['attention_mask']]) >= 0.5, 1, 0)
            m,r = report(y_test, pred, ans_class)
            awx_confusion_matrix(y_test, pred, label, ans_class)
        else:
            pred = model.predict([X_test_token['input_ids'], X_test_token['attention_mask']])
            y_pred_distilbert = np.zeros_like(pred)
            y_pred_distilbert[np.arange(len(y_pred_distilbert)), pred.argmax(1)] = 1
        
            m,r = report(y_test, y_pred_distilbert, ans_class, True)
            awx_confusion_matrix(y_test.argmax(axis=1), y_pred_distilbert.argmax(axis=1), label, ans_class) 
    
    
    # GPT-2 MODEL
    if ans_models == 'gpt':
        label = 'GPT-2 Model'
        headers=['Text','Sentiment']
        
        if ans_dataset == 'politics':
            dataset = pd.read_csv('Politics/train2682_test810/test_set_politics810.csv', sep=',', names=headers)
        else:
            dataset = pd.read_csv('Skroutz/test_set_skroutz1966.csv', sep=',', names=headers)
    
        if ans_dataset == 'politics' and ans_class == 'binary':
            dataset = delete_neutral(dataset)
        
        if ans_stopword_lemma == 'yes':
            dataset = stopwords_lemmatize(dataset)
        
        if ans_class == 'binary':
            dataset.Sentiment.replace("Positive", 1, inplace = True)
            dataset.Sentiment.replace("Negative", 0, inplace = True)
            y_test = dataset['Sentiment']
            X_test = dataset['Text']
        else:
            ohe = preprocessing.OneHotEncoder()
            y_test = ohe.fit_transform(np.array(dataset['Sentiment']).reshape(-1, 1)).toarray()
            X_test = dataset['Text']
        
        tokenizer = GPT2Tokenizer.from_pretrained("lighteternal/gpt2-finetuned-greek", from_pt=True)
        X_test_token = get_tokens_GPT(X_test, tokenizer, MAX_LENGTH)
        
        # Ορίζουμε το μοντέλο GPT-2 ως υποκλάση της Model
        class MyGPTModel(tf.keras.Model):
            def __init__(self):
                super(MyGPTModel, self).__init__()
                self.gpt = TFAutoModelForCausalLM.from_pretrained('lighteternal/gpt2-finetuned-greek', from_pt=True)
                self.flatten = tf.keras.layers.Flatten()
                
                if ans_class == 'binary':
                    self.dense1 = tf.keras.layers.Dense(1, activation='sigmoid')
                else:
                    self.dense1 = tf.keras.layers.Dense(3, activation='softmax')

            def call(self, inputs, **kwargs):
                input_ids = inputs
                outputs = self.gpt(input_ids, **kwargs)[0][:, -1, :]
                outputs = self.flatten(outputs)
                outputs = self.dense1(outputs)
                return outputs
        
        model = MyGPTModel()
        input_ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="input_ids")
        outputs = model(input_ids)
        
        # Φόρτωση των βαρών του προ-εκπαιδευμένου μοντέλου
        if ans_dataset == 'politics':
            if ans_class == 'binary' and ans_stopword_lemma == 'yes':
                model.load_weights('E:\\h5\\Politics\\train2682_test810\\train_gpt_bin_politics_with_sl.h5')
            elif ans_class == 'binary' and ans_stopword_lemma == 'no':
                model.load_weights('E:\\h5\\Politics\\train2682_test810\\train_gpt_bin_politics_without_sl.h5')
            elif ans_class == 'multi' and ans_stopword_lemma == 'yes':
                model.load_weights('E:\\h5\\Politics\\train2682_test810\\train_gpt_multi_politics_with_sl.h5')
            elif ans_class == 'multi' and ans_stopword_lemma == 'no':
                model.load_weights('E:\\h5\\Politics\\train2682_test810\\train_gpt_multi_politics_without_sl.h5')
        
        if ans_dataset == 'skroutz':
            if ans_stopword_lemma == 'yes':
                model.load_weights('E:\\h5\\Skroutz\\train_gpt_bin_skroutz_with_sl.h5')
            else:
                model.load_weights('E:\\h5\\Skroutz\\train_gpt_bin_skroutz_without_sl.h5')
        
        # Χρήση μεθόδου predict()
        if ans_class == 'binary':
            pred = np.where(model.predict([X_test_token['input_ids']]) >= 0.5, 1, 0)
            m,r = report(y_test, pred, ans_class)
            awx_confusion_matrix(y_test, pred, label, ans_class)
        else:
            pred = model.predict([X_test_token['input_ids']])
            y_pred_gpt = np.zeros_like(pred)
            y_pred_gpt[np.arange(len(y_pred_gpt)), pred.argmax(1)] = 1
        
            m,r = report(y_test, y_pred_gpt, ans_class, True)
            awx_confusion_matrix(y_test.argmax(1), y_pred_gpt.argmax(1), label, ans_class)
    return r, m    
        
# Μέθοδος get_tokens για ανάκτηση αριθμητικών χαρακτηριστικών
def get_tokens(samples, tokenizer, MAX_LENGTH):
    my_dict = tokenizer(text = list(samples),
                  add_special_tokens = True,
                  max_length = MAX_LENGTH,
                  truncation = True,
                  padding = 'max_length',
                  return_tensors = 'tf',
                  return_token_type_ids = False,
                  return_attention_mask = True,
                  verbose = True)
    return my_dict        


# Δήλωση της get_tokens_GPT μεθόδου για GPT-2.
def get_tokens_GPT(samples, tokenizer, MAX_LENGTH):
    my_dict = tokenizer(text = list(samples),   
                  add_special_tokens = False,
                  max_length = MAX_LENGTH,
                  truncation = True,      
                  padding = 'max_length',
                  return_tensors = 'tf',
                  verbose = False)
    return my_dict
        



    
    
    

