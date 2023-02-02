import streamlit as st
import pandas as pd
import numpy as np
import joblib

path = 'emotional_classic.pkl'
data = joblib

pipe_lr = joblib.load(open(path,'rb'))

def predict_emotions(docx):
    result = pipe_lr.predict([docx])
    return result[0]

def get_predict_proba(docx):
    result = pipe_lr.predict_proba([docx])
    return result[0]


def main():
    st.title('Emotion classifier app')
    menu = ['Home', 'Monitor', 'About']
    choice = st.sidebar.selectbox('Menu',menu)
    if choice == 'Home':
        st.subheader('home-emotion in text')
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area('type here')
            submit_text = st.form_submit_button(label='submit')
        if submit_text:
            col1,col2 = st.columns(2)
            prediction = predict_emotions(raw_text)
            probility = get_predict_proba(raw_text)


            with col1:
                st.success('Original Text')
                st.write(raw_text)
                st.success('Prediction')
                st.write(prediction)
            with col2:
                st.success('Prediction Probability')
                st.write(probility)
                proba_df = pd.DataFrame(probility,columns = pipe_lr.classes_)
                st.write(proba_df)

    elif choice == 'Monitor':
        st.subheader('Monitor app')

    else:
        st.subheader('About')

if __name__ == '__main__':
    main()
