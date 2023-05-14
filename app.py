import os
import streamlit as st
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain import PromptTemplate


os.environ["OPENAI_API_KEY"] = "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"

def load_file(file_path):
    loader = TextLoader(file_path)
    loaded_documents = loader.load()
    return loaded_documents

def split_document(document):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(document)
    return documents

def vector_store(documents):
    embeddings = OpenAIEmbeddings()
    vecstore = Chroma.from_documents(documents, embeddings)
    return vecstore

def prompt_template():
    template = """
you are an agent that work in a company which have data of the projects that the company did with the technologies that they used, try to be like a seller but in a nice way 
and convince him to choose our company,
i want you to find if the following text that i will give you now is similar like the projects we did if yes 
you tell him about the project and the technologies that we used and a small breif about the projects ,
try to convince him to choose us and small brief about it. we are a big company specialized in software projects,
if no you tell him about the company and try to convince him to choose us . and here is the text that you search about: 

Text:{query}

if you can't answer the question tell him about the company
Context:  we are a big company specialized in software projects
"""

    prompt = PromptTemplate(
        input_variables=["query"],
        template=template,
    )
    return prompt


def qa_retrieval(vectorstore):
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0,model_name="gpt-3.5-turbo"), 
        chain_type="stuff", 
        retriever=vectorstore.as_retriever(),
    )
    return qa


def query(q,qa,prompt):
    
    output = qa.run(prompt.format(query=q))
    return output

loadFile = load_file("test.txt")

splittedDocuments = split_document(document=loadFile)

vecstore = vector_store(documents=splittedDocuments)

template = prompt_template()

qa = qa_retrieval(vectorstore=vecstore)

st.set_page_config(page_title="Chatbot",)

st.title('Software Company ChatBot')

input = st.text_input('Enter your text')

if input:
    response = query(q=input,qa=qa,prompt=template)
    st.write(response)