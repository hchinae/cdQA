import streamlit as st  # type: ignore
###############################################
import os
import pandas as pd
from ast import literal_eval

from cdqa.utils.converters import pdf_converter
from cdqa.pipeline import QAPipeline
from cdqa.utils.download import download_model
################################################
import joblib
##################################################   
   
# Why two times we have: dump dump load load? check it

@st.experimental_memo
def get_cdqa_pipeline(bookname: str) -> QAPipeline:
    download_model(model='bert-squad_1.1', dir='./models')

    df = pdf_converter(directory_path='./books' + bookname)
    df.head()

    #pd.set_option('display.max_colwidth', None)
    df.head()

    cdqa_pipeline = QAPipeline(reader='./models/bert_qa.joblib', max_df=1.0)

    cdqa_pipeline.fit_retriever(df=df)

    print('start to joblib dumb')
    joblib.dump(cdqa_pipeline, './models/bert_qa_custom.joblib')

    print('start to joblib load')
    cdqa_pipeline=joblib.load('./models/bert_qa_custom.joblib')

    return cdqa_pipeline


def get_predictions_cdqa_pipeline(cdqa_pipeline, query: str):    
    prediction = cdqa_pipeline.predict(query, 10)
    return prediction


def format_text(paragraph: str, start_idx: int, end_idx: int) -> str:
    return (
        paragraph[:start_idx]
        + "**"
        + paragraph[start_idx:end_idx]
        + "**"
        + paragraph[end_idx:]
    )


if __name__ == "__main__":
    """
    # Closed Question Answering on (Text) Books: Rich Dad, Poor Dad
    """

    #book_to_index = st.text_input("WHICH book to index", "")

    book_to_index=''
    cdqa_pipeline = get_cdqa_pipeline(book_to_index)

    old_question = '' 
    question = st.text_input("QUESTION", "")
    if question != '' and question != old_question:    
        try:
            print('query: {}'.format(question))
            prediction = get_predictions_cdqa_pipeline(cdqa_pipeline, question)    
            print('answer1: {}'.format(prediction[0]))
            
            #answer = ''
            
            # for i in range(5):
            #     answer_part = prediction[i][0] + '\n' + prediction[i][2] + '\n' + prediction[i][3]
            #     answer += '\n'

            st.success(question)
            st.success(prediction)
            old_question = question
            
        except Exception as e:
            print(e)
            st.warning("There is an error")
