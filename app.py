import streamlit as st
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from PIL import Image


filename = 'nlp_model.pkl'
clf = pickle.load(open(filename,'rb'))
cv = pickle.load(open('tranform.pkl','rb'))

html_temp = """<div><center><p style="font-size:16px; color:grey">&copy;2020 Anmol</p></center></div>"""

choice = st.sidebar.selectbox("Select",["-","Why","Check"])
st.sidebar.write("---")
st.sidebar.markdown(html_temp, unsafe_allow_html=True)
st.markdown('''<h1><center>Spam or Ham</center></h1>''',unsafe_allow_html=True)
st.write('')
st.text('')

if choice == "-":
    st.write('\n')
    im = """<div><center><img src="https://user-images.githubusercontent.com/51512071/98292903-7ebfed80-1fd3-11eb-9951-036549e56ab5.jpg" alt="Spam or Ham" style="max-width:100%;width:100%"></center></div>"""
    st.markdown(im, unsafe_allow_html=True)

    
if choice == "Why":
    ima = """<div><center><img src="https://user-images.githubusercontent.com/51512071/98277331-41e8fc00-1fbd-11eb-9d94-09d37b71efdb.jpg" alt="Spam or Ham" style="max-width:100%;width:100%"></center></div>"""
    st.markdown(ima, unsafe_allow_html=True)

    about1 = """<div style="align-items:center; background-color: #F5F5F5; padding:15px"><h1>Why</h1><p style = "font-size:18px;">As we all encounter various kinds of mails everyday in our inbox but all of them are not what we desire or wish to land in our inbox, so here's a simple demo of how easy it is to sort the so called <b>Spam</b> (not required mails) from a <b>Ham</b> (Desired mail)</p><br></div>"""

    st.markdown(about1, unsafe_allow_html=True)

if choice == "Check":
    message = st.text_area("Enter Message to check",max_chars=500,height=180, )
    ch = st.button('Check')
    data = [message]
    vect = cv.transform(data).toarray()
    my_prediction = clf.predict(vect)

    st.write('\n')
    # <img src="./check.jpg" alt="Spam or Ham" style="max-width:100%; width:100%">

    # im = Image.open('check.jpg')
    # st.image(im,width=600)


    st.write('\n')

    c1, c2 = st.beta_columns(2)
    with c1:
        st.subheader('Prediction:')
    with c2:
        if ch == True and message != "":
            if my_prediction == 1:
                st.header("It's a Spam")
            else:
                st.header("Just a ham")

    st.write('\n')
    st.write('\n')
    st.write('\n')

    


def predict():
     df= pd.read_csv("spam.csv", encoding="latin-1")
     df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
     # Features and Labels
     df['label'] = df['class'].map({'ham': 0, 'spam': 1})
     X = df['message']
     y = df['label']
 	
     #Extract Feature With CountVectorizer
     cv = CountVectorizer()
     X = cv.fit_transform(X) # Fit the Data
 
     pickle.dump(cv, open('tranform.pkl', 'wb'))
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

 	 #Naive Bayes Classifier
# =============================================================================
#     from sklearn.naive_bayes import MultinomialNB
      
     clf = MultinomialNB()
     clf.fit(X_train,y_train)
     clf.score(X_test,y_test)
     filename = 'nlp_model.pkl'
     pickle.dump(clf, open(filename, 'wb'))
