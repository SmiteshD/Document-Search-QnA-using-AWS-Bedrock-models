# ChatWithPDFs-Bedrock

**ChatWithPDFs-Bedrock** is a Streamlit-based application that allows users to interact with PDF documents using AWS Bedrock. It combines the power of embeddings, vector search, and language models to provide concise and context-aware answers to user queries.

---

## Features

- **Upload PDF Files**: Load multiple PDFs and process their content for querying.
- **AWS Bedrock Integration**: Uses Amazon Bedrock for embeddings and language model responses.
- **Efficient Vector Search**: Built with FAISS for fast similarity-based retrieval.
- **Custom Prompts**: Generate detailed and concise answers to user questions.
- **Streamlit Interface**: Simple and interactive user interface for seamless use.

---

## How It Works

1. **Data Ingestion**: Upload PDFs and split them into manageable text chunks.
2. **Vectorization**: Generate embeddings and store them in a FAISS index.
3. **Query Handling**: Ask questions about the content, and the app retrieves and processes relevant information to provide an answer.
4. **LLM Response**: Uses AWS Bedrock's Llama3 model to generate concise and context-aware responses.

---

## Requirements

- Python 3.10 or higher
- AWS credentials with access to Bedrock
- Dependencies listed in `requirements.txt`
---

## Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ChatWithPDFs-Bedrock.git
   cd ChatWithPDFs-Bedrock
   
2. Install dependencies:
  ```bash
   pip install -r requirements.txt
  ```
3. Set up AWS credentials in your environment.

4. Run the app:
  ```bash
   streamlit run app.py
  ```

##Usage

- Add your PDF files to the Data/ directory.
- Start the app and update the vector store via the Vectors Update button in the sidebar.
- Enter your question in the input box and click Output to retrieve answers.
