import os
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from rich import print
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from sentence_transformers import CrossEncoder
from reranking_models import reranking_cohere, reranking_colbert, reranking_gpt
from contextual_retrieval import ContextualRetrieval, contextualize_chunks_for_retrieval

# Keeping existing code from original rag.py
import fitz  # PyMuPDF
from langchain.docstore.document import Document

def load_pdf(files="data/2306.02707.pdf"):
    """
    Loads documents from PDF files using PyMuPDF.

    Parameters:
    - files: A string representing a single file path or a list of strings representing multiple file paths.

    Returns:
    - A list of Document objects loaded from the provided PDF files.

    Raises:
    - FileNotFoundError: If any of the provided file paths do not exist.
    - Exception: For any other issues encountered during file loading.

    The function applies post-processing steps such as cleaning extra whitespace and grouping broken paragraphs.
    """
    if not isinstance(files, list):
        files = [files]  # Ensure 'files' is always a list

    documents = []
    for file_path in files:
        try:
            # Open the PDF file
            doc = fitz.open(file_path)
            text = ""
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text("text")

            # Apply post-processing steps
            text = clean_extra_whitespace(text)
            text = group_broken_paragraphs(text)

            # Create a Document object
            document = Document(
                page_content=text,
                metadata={"source": file_path}
            )
            documents.append(document)
        except FileNotFoundError as e:
            print(f"File not found: {e.filename}")
            raise
        except Exception as e:
            print(f"An error occurred while loading {file_path}: {e}")
            raise

    return documents

def clean_extra_whitespace(text):
    """
    Cleans extra whitespace from the provided text.

    Parameters:
    - text: A string representing the text to be cleaned.

    Returns:
    - A string with extra whitespace removed.
    """
    return ' '.join(text.split())

def group_broken_paragraphs(text):
    """
    Groups broken paragraphs in the provided text.

    Parameters:
    - text: A string representing the text to be processed.

    Returns:
    - A string with broken paragraphs grouped.
    """
    return text.replace("\n", " ").replace("\r", " ")


def split_documents(
    chunk_size: int,
    knowledge_base,
    tokenizer_name= "openai",
):
    """
    Splits documents into chunks of maximum size `chunk_size` tokens, using a specified tokenizer.
    
    Parameters:
    - chunk_size: The maximum number of tokens for each chunk.
    - knowledge_base: A list of LangchainDocument objects to be split.
    - tokenizer_name: (Optional) The name of the tokenizer to use. Defaults to `EMBEDDING_MODEL_NAME`.
    
    Returns:
    - A list of LangchainDocument objects, each representing a chunk. Duplicates are removed based on `page_content`.
    
    Raises:
    - ImportError: If necessary modules for tokenization are not available.
    """

    if tokenizer_name=="openai":
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            separators=["\n\n\n", "\n\n", "\n", ".", ""],
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            model_name="gpt-4",
            is_separator_regex=False,
            add_start_index=True,
            strip_whitespace=True,
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(tokenizer_name),
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
        )

    docs_processed = (text_splitter.split_documents([doc]) for doc in knowledge_base)
    # Flatten list and remove duplicates more efficiently
    unique_texts = set()
    docs_processed_unique = []
    for doc_chunk in docs_processed:
        for doc in doc_chunk:
            if doc.page_content not in unique_texts:
                unique_texts.add(doc.page_content)
                docs_processed_unique.append(doc)

    return docs_processed_unique


def load_embedding_model(
    model_name = "openai", 
):
    """
    Loads an embedding model from the Hugging Face repository with specified configurations.

    Parameters:
    - model_name: The name of the model to load. Defaults to "BAAI/bge-large-en-v1.5".
    - device: The device to run the model on (e.g., 'cpu', 'cuda', 'mps'). Defaults to 'mps'.

    Returns:
    - An instance of HuggingFaceBgeEmbeddings configured with the specified model and device.

    Raises:
    - ValueError: If an unsupported device is specified.
    - OSError: If the model cannot be loaded from the Hugging Face repository.
    """

    try:
        if model_name=="openai":
             embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            model_name="BAAI/bge-large-en-v1.5"
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            model_kwargs = {"device": device}
            encode_kwargs = {"normalize_embeddings": True}  # For cosine similarity computation

            embedding_model = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        return embedding_model
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        raise


def retrieve_context(query, retriever):
    """
    Retrieves and reranks documents relevant to a given query.

    Parameters:
    - query: The search query as a string.
    - retriever: An instance of a Retriever class used to fetch initial documents.

    Returns:
    - A list of reranked documents deemed relevant to the query.

    """
    retrieved_docs = retriever.get_relevant_documents(query)

    return retrieved_docs


def get_ensemble_retriever(docs, embedding_model, collection_name="test", top_k=3):
    """
    Initializes a retriever object to fetch the top_k most relevant documents based on cosine similarity.

    Parameters:
    - docs: A list of documents to be indexed and retrieved.
    - embedding_model: The embedding model to use for generating document embeddings.
    - top_k: The number of top relevant documents to retrieve. Defaults to 3.

    Returns:
    - A retriever object configured to retrieve the top_k relevant documents.

    Raises:
    - ValueError: If any input parameter is invalid.
    """

    # Hybrid search
    # Example of parameter validation (optional)
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    try:
        vector_store = Chroma.from_documents(
            docs, 
            embedding_model,
            collection_name=collection_name,
        )

        retriever = vector_store.as_retriever(search_kwargs={"k":top_k})
        # retriever.k = top_k

        # add keyword search 
        keyword_retriever = BM25Retriever.from_documents(docs)
        keyword_retriever.k =  3

        ensemble_retriever = EnsembleRetriever(retrievers=[retriever,
                                                    keyword_retriever],
                                        weights=[0.5, 0.5])

        return ensemble_retriever
    except Exception as e:
        print(f"An error occurred while initializing the retriever: {e}")
        raise


def create_multi_query_retriever(base_retriever, llm):
    """
    Create a multi-query retriever based on the base retriever and LLM. 

    Parameters: 
    - base_retriever: base retriever 
    - llm: LLM to generate variations of queries.

    Returns: 
    - A retriever that is able to generate multiple queries. 
    """

    multiquery_retriever = MultiQueryRetriever.from_llm(base_retriever, llm)

    return multiquery_retriever


def create_parent_retriever(
    docs, 
    embeddings_model,
    collection_name="split_documents", 
    top_k=5,
    persist_directory=None,
    use_contextual_retrieval=False,
    llm=None
):
    
    """
    Initializes a retriever object to fetch the top_k most relevant documents based on cosine similarity.

    Parameters:
    - docs: A list of documents to be indexed and retrieved.
    - embedding_model: The embedding model to use for generating document embeddings.
    - collection_name: The name of the collection
    - top_k: The number of top relevant documents to retrieve. Defaults to 3.
    - persist_directory: directory where you want to store your vectorDB
    - use_contextual_retrieval: Whether to use contextual retrieval to enhance chunks
    - llm: Language model to use for contextual retrieval (if enabled)

    Returns:
    - A retriever object configured to retrieve the top_k relevant documents.

    Raises:
    - ValueError: If any input parameter is invalid.
    """
        
    parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n\n\n", "\n\n", "\n", ".", ""],
        chunk_size=512,
        chunk_overlap=0,
        model_name="gpt-4",
        is_separator_regex=False,
    )

    # This text splitter is used to create the child documents
    child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n\n\n", "\n\n", "\n", ".", ""],
        chunk_size=256,
        chunk_overlap=0,
        model_name="gpt-4",
        is_separator_regex=False,
    )

    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings_model,
        persist_directory=persist_directory,
    )

    # The storage layer for the parent documents
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        k=10,
    )
    
    # If contextual retrieval is enabled, we'll need to process documents differently
    if use_contextual_retrieval:
        processed_docs = []
        for doc in docs:
            # First split the document into chunks for processing
            parent_chunks = parent_splitter.split_documents([doc])
            if not parent_chunks:
                continue
                
            # For each parent chunk, apply child splitting to get smaller chunks
            for parent_chunk in parent_chunks:
                child_chunks = child_splitter.split_documents([parent_chunk])
                
                # Now contextualize these child chunks
                contextualizer = ContextualRetrieval(llm=llm)
                contextualized_chunks = contextualizer.contextualize_chunks(
                    document_text=doc.page_content,
                    chunks=child_chunks
                )
                
                # Create a new parent document with the contextualized chunks
                parent_chunk.metadata["chunks"] = contextualized_chunks
                processed_docs.append(parent_chunk)
        
        # Add the processed documents to the retriever
        retriever.add_documents(processed_docs)
    else:
        # Standard approach without contextual retrieval
        retriever.add_documents(docs)

    return retriever


def rerank_docs(query, retrieved_docs, reranker_model):
    """
    Rerank the provided context chunks

    Parameters:
    - reranker_model: the model selection.
    - query: user query - string
    - retrieved_docs: chunks that needs to be ranked. 

    
    Returns:
    - Sorted list of chunks based on their relevance to the query. 

    """

    if reranker_model=="gpt":
        ranked_docs = reranking_gpt(retrieved_docs, query)
    elif reranker_model=="cohere":
        ranked_docs = reranking_cohere(retrieved_docs, query)
    elif reranker_model=="colbert":
        ranked_docs = reranking_colbert(retrieved_docs, query)
    else: # just return the original order
        ranked_docs = [(query, r.page_content) for r in retrieved_docs]

    return ranked_docs


def retrieve_context_reranked(query, retriever, reranker_model="gpt4"):
    """
    Retrieve the context and rerank them based on the selected re-ranking model.

    Parameters:
    - query: user query - string
    - retrieved_docs: chunks that needs to be ranked. 
    - reranker_model: the model selection.

    
    Returns:
    - Sorted list of chunks based on their relevance to the query. 

    """

    retrieved_docs = retriever.get_relevant_documents(query)

    if len(retrieved_docs) == 0:
        print(
            f"Couldn't retrieve any relevant document with the query `{query}`. Try modifying your question!"
        )
    reranked_docs = rerank_docs(
        query=query, retrieved_docs=retrieved_docs, reranker_model=reranker_model
    )
    return reranked_docs


# New functions for contextual retrieval

# def create_contextual_retriever(
#     docs,
#     embeddings_model,
#     llm,
#     collection_name="contextual_documents",
#     top_k=5,
#     persist_directory=None
# ):
#     """
#     Creates a retriever that uses Anthropic's Contextual Retrieval approach.
    
#     Parameters:
#     - docs: A list of documents to be indexed and retrieved.
#     - embeddings_model: The embedding model for generating document embeddings.
#     - llm: Language model to use for generating contextual information.
#     - collection_name: The name of the collection in the vector store.
#     - top_k: The number of top relevant documents to retrieve.
#     - persist_directory: Directory to persist the vector store.
    
#     Returns:
#     - A retriever configured with contextual embeddings.
#     """
#     # First, we need to process all documents with our contextual retriever
#     processed_docs = []
#     contextualizer = ContextualRetrieval(llm=llm)
    
#     for doc in docs:
#         # Split the document into chunks first
#         text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#             separators=["\n\n\n", "\n\n", "\n", ".", ""],
#             chunk_size=256,
#             chunk_overlap=25,
#             model_name="gpt-4",
#             is_separator_regex=False,
#         )
#         chunks = text_splitter.split_documents([doc])
        
#         # Now contextualize each chunk
#         contextualized_chunks = contextualizer.contextualize_chunks(
#             document_text=doc.page_content,
#             chunks=chunks
#         )
        
#         processed_docs.extend(contextualized_chunks)
    
#     # Create a vector store with the contextualized documents
#     vector_store = Chroma.from_documents(
#         processed_docs,
#         embeddings_model,
#         collection_name=collection_name,
#         persist_directory=persist_directory
#     )
    
#     # Create the vector retriever
#     vector_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    
#     # Create BM25 retriever with contextualized documents
#     bm25_retriever = BM25Retriever.from_documents(processed_docs)
#     bm25_retriever.k = top_k
    
#     # Create an ensemble retriever that combines both
#     ensemble_retriever = EnsembleRetriever(
#         retrievers=[vector_retriever, bm25_retriever],
#         weights=[0.7, 0.3]  # Giving more weight to vector search
#     )
    
#     return ensemble_retriever



def create_contextual_retriever(
    docs,
    embeddings_model,
    llm,
    collection_name="contextual_documents",
    top_k=5,
    persist_directory=None,
    context_window_size=3  # Number of chunks on either side for context
):
    """
    Creates a retriever that uses Anthropic's Contextual Retrieval approach with chunk-based context windows.
    
    Parameters:
    - docs: A list of documents to be indexed and retrieved.
    - embeddings_model: The embedding model for generating document embeddings.
    - llm: Language model to use for generating contextual information.
    - collection_name: The name of the collection in the vector store.
    - top_k: The number of top relevant documents to retrieve.
    - persist_directory: Directory to persist the vector store.
    - context_window_size: Number of chunks on either side to include in the context window.
    
    Returns:
    - A retriever configured with contextual embeddings.
    """
    # First, we need to process all documents with our contextual retriever
    processed_docs = []
    
    for doc in docs:
        # Split the document into chunks first
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            separators=["\n\n\n", "\n\n", "\n", ".", ""],
            chunk_size=256,
            chunk_overlap=25,
            model_name="gpt-4",
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents([doc])
        
        # Add document source to all chunks' metadata
        for chunk in chunks:
            if "source" not in chunk.metadata and "source" in doc.metadata:
                chunk.metadata["source"] = doc.metadata["source"]
        
        print(f"Document '{doc.metadata.get('source', 'unknown')}' split into {len(chunks)} chunks")
        
        # Only process if we have chunks
        if chunks:
            # Now contextualize the chunks using surrounding chunks as context
            contextualizer = ContextualRetrieval(llm=llm)
            contextualized_chunks = contextualizer.contextualize_chunks(doc.page_content, chunks)
            
            processed_docs.extend(contextualized_chunks)
    
    print(f"Created {len(processed_docs)} contextualized chunks from {len(docs)} documents")
    
    # Create a vector store with the contextualized documents
    vector_store = Chroma.from_documents(
        processed_docs,
        embeddings_model,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    # Create the vector retriever
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    
    # Create BM25 retriever with contextualized documents
    bm25_retriever = BM25Retriever.from_documents(processed_docs)
    bm25_retriever.k = top_k
    
    # Create an ensemble retriever that combines both
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]  # Giving more weight to vector search
    )
    
    return ensemble_retriever