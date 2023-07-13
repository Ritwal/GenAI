from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from embedding import CHROMA_SETTINGS

load_dotenv()

embeddings_model_name = "all-MiniLM-L6-v2" 
persist_directory = "db" 
model_path = "models/ggml-gpt4all-j-v1.3-groovy.bin" 
model_n_ctx = 1000 
target_source_chunks = 4 



def main():
        
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    
    # Prepare the Gen model
    llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj',verbose=False)
        
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain
        res = qa(query)
        answer = res['result']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

if __name__ == "__main__":
    main()
