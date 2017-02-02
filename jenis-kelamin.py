import sys, argparse, pickle, os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

# main
def main(args):
    if(args.ml == 'LG'):
        result = predict_lg(args.name, args.train)
        ml_type = 'Logistic Regression'
    elif(args.ml == 'RF'):
        result = predict_rf(args.name, args.train)
        ml_type = 'Random Forest'
    else:
        result = predict_nb(args.name, args.train)
        ml_type = 'Naive Bayes'
    
    
    print ("Prediksi jenis kelamin dengan", ml_type, ":")
    jk_label = {1:"Pria", 0:"Wanita"}
    print(args.name, ' : ', jk_label[result])

# load dataset
def load_data(dataset="./data/data-pemilih-kpu.csv"):
    df = pd.read_csv(dataset, encoding = 'utf-8-sig')
    df = df.dropna(how='all')
    
    jk_map = {"Laki-Laki" : 1, "Perempuan" : 0}
    df["jenis_kelamin"] = df["jenis_kelamin"].map(jk_map)

    feature_col_names = ["nama"]
    predicted_class_names = ["jenis_kelamin"]
    X = df[feature_col_names].values     
    y = df[predicted_class_names].values 
    
    #split train:test data 70:30
    split_test_size = 0.30
    text_train, text_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, stratify=y, random_state=42) 
    
    return (text_train, text_test, y_train, y_test)

# Naive Bayes implementation
def predict_nb(name, dataset):
    if os.path.isfile("./data/pipe_nb.pkl") and dataset is None:        
        file_nb = open('./data/pipe_nb.pkl', 'rb')
        pipe_nb = pickle.load(file_nb)
    else:
        file_nb = open('./data/pipe_nb.pkl', 'wb')
        pipe_nb = Pipeline([('vect', CountVectorizer(analyzer = 'char_wb', ngram_range=(2,6))),
                            ('tfidf', TfidfTransformer()),
                            ('clf', MultinomialNB())])       
        #train and dump to file                     
        dataset = load_data(dataset)
        pipe_nb = pipe_nb.fit(dataset[0].ravel(), dataset[2].ravel())
        pickle.dump(pipe_nb, file_nb)
        
        #Akurasi
        predicted = pipe_nb.predict(dataset[1].ravel())
        Akurasi = np.mean(predicted == dataset[3].ravel())*100
        print("Akurasi :", Akurasi, "%")
    
    return pipe_nb.predict([name])[0]

# Logistic Regression implementation
def predict_lg(name, dataset):
    if os.path.isfile("./data/pipe_lg.pkl") and dataset is None:        
        file_lg = open('./data/pipe_lg.pkl', 'rb')
        pipe_lg = pickle.load(file_lg)
    else:
        file_lg = open('./data/pipe_lg.pkl', 'wb')
        pipe_lg = Pipeline([('vect', CountVectorizer(analyzer = 'char_wb', ngram_range=(2,6))),
                            ('tfidf', TfidfTransformer()),
                            ('clf', LogisticRegression())])        
        dataset = load_data(dataset)
        pipe_lg = pipe_lg.fit(dataset[0].ravel(), dataset[2].ravel())
        pickle.dump(pipe_lg, file_lg)

        #Akurasi
        predicted = pipe_lg.predict(dataset[1].ravel())
        Akurasi = np.mean(predicted == dataset[3].ravel())*100
        print("Akurasi :", Akurasi, "%")
    
    return pipe_lg.predict([name])[0]

# Random Forest implementation
def predict_rf(name, dataset):
    if os.path.isfile("./data/pipe_rf.pkl") and dataset is None:         
        file_rf = open('./data/pipe_rf.pkl', 'rb')
        pipe_rf = pickle.load(file_rf)
    else:
        file_rf = open('./data/pipe_rf.pkl', 'wb')
        pipe_rf = Pipeline([('vect', CountVectorizer(analyzer = 'char_wb', ngram_range=(2,6))),
                            ('tfidf', TfidfTransformer()),
                            ('clf', RandomForestClassifier(n_estimators=10, n_jobs=-1))])        
        dataset = load_data(dataset)
        pipe_rf = pipe_rf.fit(dataset[0].ravel(), dataset[2].ravel())
        pickle.dump(pipe_rf, file_rf)

        #Akurasi
        predicted = pipe_rf.predict(dataset[1].ravel())
        Akurasi = np.mean(predicted == dataset[3].ravel())*100
        print("Akurasi :", Akurasi, "%")
    
    return pipe_rf.predict([name])[0]

# args setting
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = "Menentukan jenis kelamin berdasarkan nama Bahasa Indoensia")
 
  parser.add_argument(
                      "name",
                      help = "Nama",
                      metavar='nama'
                      )
  parser.add_argument(
                      "-ml",
                      help = "NB=Naive Bayes(default); LG=Logistic Regression; RF=Random Forest",
                      choices=["NB", "LG", "RF"]
                      )
  parser.add_argument(
                      "-t",
                      "--train",
                      help="Training ulang dengan dataset yang ditentukan")
  args = parser.parse_args()
  
  main(args)