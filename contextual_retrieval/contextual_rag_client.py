"""
RAG client that uses Anthropic's Contextual Retrieval approach.
"""

from langchain.callbacks import FileCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from loguru import logger
import os

from rag import (
    load_embedding_model,
    load_pdf,
    retrieve_context_reranked
)
from rag_contextual import create_contextual_retriever


class ContextualRagClient:
    """
    A RAG client that uses Anthropic's Contextual Retrieval approach.
    """
    
    def __init__(self, files, use_anthropic=True, use_openai=True):
        """
        Initialize the contextual RAG client.
        
        Args:
            files: Path(s) to the document file(s) to use.
            use_anthropic: Whether to use Anthropic's Claude for the LLM.
            use_openai: Whether to use OpenAI for embeddings.
        """
        # Load embedding model
        if use_openai:
            self.embedding_model = load_embedding_model(model_name="openai")
        else:
            self.embedding_model = load_embedding_model(model_name="BAAI/bge-large-en-v1.5")
        
        # Load documents
        docs = load_pdf(files=files)
        
        # Initialize the LLM for contextual retrieval and RAG
        if use_anthropic and os.environ.get("ANTHROPIC_API_KEY"):
            self.llm = ChatAnthropic(model="claude-3-haiku-20240307")
        else:
            self.llm = ChatOpenAI(model_name="gpt-4.1")
        
        # Create the contextual retriever
        self.retriever = create_contextual_retriever(
            docs=docs,
            embeddings_model=self.embedding_model, 
            llm=self.llm,
            collection_name="contextual_docs"
        )
        
        # Create the prompt template for RAG
        prompt_template = ChatPromptTemplate.from_template(
            (
                """
                You are a helpful AI Assistant. Your job is to extract information from the provided CONTEXT based on the user question.
                Think step by step and only use the information from the CONTEXT that is relevant to the user question. Provide detailed responses.   

                QUESTION: ```{question}```\n
                CONTEXT: ```{context}```\n
                """
            )
        )
        
        # Create the RAG chain
        self.chain = prompt_template | self.llm | StrOutputParser()
    
    def stream(self, query):
        """
        Stream responses from the RAG system for a given query.
        
        Args:
            query: The user's query.
            
        Yields:
            Streamed responses from the LLM.
        """
        try:
            context_list = self.retrieve_context_reranked(query)
            logger.info(f"Retrieved context: {context_list}")
            
            # Combine the top contexts
            context = ""
            for i, cont in enumerate(context_list):
                if i < 3:  # Use top 3 contexts
                    context = context + "\n" + cont
                else:
                    break
            
            logger.info(f"Combined context: {context}")
        except Exception as e:
            context = e.args[0]
            logger.error(f"Error retrieving context: {e}")
        
        # Stream the response
        for r in self.chain.stream({"context": context, "question": query}):
            yield r
    
    def retrieve_context_reranked(self, query):
        """
        Retrieve and rerank context for a query.
        
        Args:
            query: The user's query.
            
        Returns:
            A list of reranked context chunks.
        """
        return retrieve_context_reranked(
            query, 
            retriever=self.retriever, 
            reranker_model="cohere"  # You can use "gpt" or "colbert" for local model
        )
    
    def generate(self, query):
        """
        Generate a response for a query.
        
        Args:
            query: The user's query.
            
        Returns:
            A dictionary with the context and response.
        """
        contexts = self.retrieve_context_reranked(query)
        
        # Combine the top contexts
        text = ""
        for i, cont in enumerate(contexts):
            if i < 3:  # Use top 3 contexts
                text = text + "\n" + cont
            else:
                break
        
        logger.info(f"Combined context: {text}")
        
        # Generate the response
        return {
            "contexts": text,
            "response": self.chain.invoke(
                {"context": text, "question": query}
            ),
        }
