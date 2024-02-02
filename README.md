# QnA_RAG_Pipelines

- It required a Hugging Face API token to run this pipeline, which we can get from https://huggingface.co/docs/hub/security-tokens.
- Once we have tokens, update the 'retrieval.py' file with your hugging face API token key.
- We're using the Mixtral-8x7B-Instruct-v0.1 Large Language Model (LLM) for the generation part. Learn more about it at https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1.
The pipeline uses the Chroma vector store from Langchain. For more on vector stores, visit https://python.langchain.com/docs/modules/data_connection/vectorstores/.
- Add pdf documents to the "pdf_data" folder for the RAG pipeline, and remember, Currently it reads PDF files only.
- Make sure the command installer have updated one. i.e; bash, pip
- Then, install the requirements by using 'pip install -r requirements.txt'.
- Install Chromadb
- Run the code by typing 'python retrieval.py'.
