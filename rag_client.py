from langchain.callbacks import FileCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from loguru import logger

from rag import (
    create_parent_retriever,
    load_embedding_model,
    load_pdf,
    retrieve_context_reranked,
)


class rag_client:
    embedding_model = load_embedding_model(model_name="openai") #model_name="BAAI/bge-large-en-v1.5"

    def __init__(self, files,):
        docs = load_pdf(files=files)
        self.retriever = create_parent_retriever(docs, self.embedding_model)

        llm = ChatOpenAI(model_name="gpt-4o")
        # llm = ChatOllama(model="llama3")

        prompt_template = ChatPromptTemplate.from_template(
            (
                """
            You are a helpful AI Assistant. Your job is to extract information from the provided CONTEXT based on the user question.
            Think step by step and only use the information from the CONTEXT that is relevant to the user question. Provide detailed responses.   

                QUESTION: ```{question}```\n
                CONTEXT: ```{context}```\n"""
            )
        )
        self.chain = prompt_template | llm | StrOutputParser()

    def stream(self, query):
        try:
            context_list = self.retrieve_context_reranked(query)
            print(f"Context: {context_list}")
            context = ""
            for i,cont in enumerate(context_list):
                if i <3:
                    context = context +"\n"+ cont
                else:
                    break
            print(context)
        except Exception as e:
            context = e.args[0]
        logger.info(context)
        for r in self.chain.stream({"context": context, "question": query}):
            yield r

    def retrieve_context_reranked(self, query):
        return retrieve_context_reranked(
            query, retriever=self.retriever, reranker_model="gpt4" # colbert for local model
        )

    def generate(self, query):
        contexts = self.retrieve_context_reranked(query)
        # print(contexts)
        text = ""
        for i,cont in enumerate(contexts):
            if i <3:
                text = text +"\n"+ cont
            else:
                break
        print(f"Here is the text: {text}")
        return {
            "contexts": text,
            "response": self.chain.invoke(
                {"context": text, "question": query}
            ),
        }