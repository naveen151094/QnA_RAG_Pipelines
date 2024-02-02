from database_created import generate_data_store
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

vectordb = generate_data_store()
retriever = vectordb.as_retriever(search_type = "similarity", search_kwargs={"k": 5})

def main(question):
    llm = HuggingFaceHub(
        #repo_id = "mistralai/Mistral-7B-Instruct-v0.1",
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        model_kwargs={"temperature":0.3, "max_length":600},
        huggingfacehub_api_token = "*************************"
    )

    print('In Search Mode')
    rqa_prompt_template = """Based on following pieces of context, answer the questions.
                        Answer only from the context. If you are unable to fetch the answer, say you do not know.
                    {context}
                    Explain in detail.
                    Question: {question}
                    """
    RQA_PROMPT = PromptTemplate(
        template = rqa_prompt_template, input_variables = ["context","question"]
    )
    rqa_chain_type_kwargs = {"prompt": RQA_PROMPT}

    qa = RetrievalQA.from_chain_type(llm,
                                     chain_type="stuff",
                                     retriever = retriever,
                                     chain_type_kwargs=rqa_chain_type_kwargs,
                                     return_source_documents = True,
                                     verbose = False)
    result = qa({"query": question})
    return result

if __name__ == "__main__":
    query = input("Ask a question: ")
    print("Wait...")
    result = main(query)
    print(result['result'])