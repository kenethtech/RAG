from loadoc import load_documents, store_to_chroma, split_text, content_retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

def generate_vectore_store():
    documents = load_documents()
    chunks = split_text(documents)
    store_to_chroma(chunks)


def agent(queries, context: list[Document]):
    llm = ChatGroq(model="openai/gpt-oss-120b",
                   groq_api_key=os.getenv("GROQ_API_KEY"),
                   temperature=0.7)

    template = PromptTemplate.from_template("""
               You are the company customer assistant agent. Answer the users queries using the provided context.
               context: {context}
               queries: {queries}
               
               If you do not know say our assistant will reply to you shortly.
               """)
    parser = StrOutputParser()

    chain = template | llm | parser

    result= chain.invoke({
        "context": context,
        "queries": queries
    })
    return result

if __name__ == "__main__":
    queries = str(input("Enter the query to search:"))
    context = content_retriever(queries)

    answer = agent(queries, context)
    print(answer)

    
