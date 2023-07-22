import os
import json
import time
import requests
import streamlit as st
from dotenv import load_dotenv
from newspaper import Article
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.schema import (
    HumanMessage
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate



def main ():
    load_dotenv()
    st.set_page_config(page_title="NEWS ARTICLE SUMMARIZER")
    st.header("NEWS ARTICLE SUMMARIZER")
    st.sidebar.title("Developed by Tomiwa Samuel")
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }

    st.sidebar.markdown('---')
    temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value =0.0)
    st.sidebar.markdown('---')
    article_url = st.sidebar.text_input('Input News URL')
    st.sidebar.markdown('---')
    translate = st.sidebar.checkbox('Translate News')  

    if article_url:

        with st.empty():

            session = requests.Session()
            try:
                response = session.get(article_url, headers = headers, timeout=10)

                if response.status_code == 200:
                    article = Article(article_url)
                    article.download()
                    article.parse()
                else:
                    print(f"failed to fetch article from {article_url}")

            except Exception as e:
                print(f"Error occurred while fetching article at {article_url}: {e}")

            # we get the article data from the scraping part
            article_title = article.title
            article_text = article.text

            # prepare template for prompt
            template = """You are a very good assistant that summarizes online articles.

            Here's the article you want to summarize.

            ==================
            Title: {article_title}

            {article_text}
            ==================

            Write a detailed summary of the previous article in a 15 bulleted list points and also stating the title of the article first. 
            """

            prompt = template.format(article_title = article_title, article_text = article_text)

            messages = [HumanMessage(content=prompt)]

            chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature = temperature)
            # generate summary
            summary = chat(messages)
            response = summary.content

            st.write(response)

            if translate:
                target_language = st.sidebar.text_input('Input Target Language')
                source_language = 'English'
                text = summary.content
                
                if target_language:

                    translation_template = "translate the following text from {source_language} to {target_language}: {text}"
                    translation_prompt = PromptTemplate(input_variables= ['source_language', 'target_language', 'text'], template = translation_template)
                    translation_chain = LLMChain(llm= chat, prompt= translation_prompt)

                    response = translation_chain.predict(source_language =source_language, target_language = target_language ,text = text)
                    st.write(response)
            


if __name__ == "__main__":
    main()