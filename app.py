import json
import boto3
import os
from langchain.llms.bedrock import Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st

##bedrock_client
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)


##Data Ingestion
def data_ingestion():
    loader=PyPDFDirectoryLoader("Data")
    documents=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    docs=text_splitter.split_documents(documents)
    return docs

##Vectorstore and embeddings
def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(docs,bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_llm():
    llm=Bedrock(model_id="meta.llama3-8b-instruct-v1:0",client=bedrock,
                model_kwargs={"prompt":"this is where you place your input text",
                              "max_gen_len":512,
                              "temperature":0.5,
                              "top_p":0.9})
    return llm

prompt_template="""
Human: Use the following oieces of context to provide a concise answer to the question at the end 
but try to summarize in 250 words with detailed explainations. If you don't know the answer just say that you don't know,
don't try to mak up the answer.
<context>
{context}
</context

Question:{question}

Assistant:
"""

PROMPT=PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k":3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt":PROMPT}
    )

    answer=qa({"query":query})
    return answer['result']

def main():
    st.header("Chat with PDF using AWS Bedrock🔍")

    user_question=st.text_input("Ask a.Question from the PDF Files")

    with st.sidebar:
        st.title("Menu:")

        if st.button("Vectors Update"):
            with st.spinner("Processing ... "):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Output"):
        with st.spinner("Processing ... "):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm=get_llm()

            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")



if __name__=="__main__":
    main()
