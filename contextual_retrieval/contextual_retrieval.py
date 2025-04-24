"""
Implementation of Anthropic's Contextual Retrieval approach.

This module provides utilities to enhance traditional RAG systems by adding context
to document chunks before they are embedded or indexed, improving retrieval accuracy.
"""

import os
from typing import List, Dict, Any, Optional
from langchain.schema.document import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from rich import print

# Default context prompt templates
DEFAULT_DOCUMENT_PROMPT = """<document>
{whole_document}
</document>"""

DEFAULT_CHUNK_PROMPT = """Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

class ContextualRetrieval:
    """
    Class for implementing Anthropic's Contextual Retrieval approach.
    
    This class adds contextual information to document chunks before embedding
    to improve retrieval accuracy in RAG systems.
    """
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        document_prompt: str = DEFAULT_DOCUMENT_PROMPT,
        chunk_prompt: str = DEFAULT_CHUNK_PROMPT
    ):
        """
        Initialize the ContextualRetrieval class.
        
        Args:
            llm: A language model for generating contextual information.
                If not provided, will try to use Anthropic's Claude if an API key is available,
                otherwise falls back to OpenAI.
            document_prompt: Template for providing the document context.
            chunk_prompt: Template for requesting context for a chunk.
        """
        self.document_prompt = document_prompt
        self.chunk_prompt = chunk_prompt
        
        # Initialize LLM if not provided
        if llm is None:
            if os.environ.get("ANTHROPIC_API_KEY"):
                self.llm = ChatAnthropic(model="claude-3-haiku-20240307")
            elif os.environ.get("OPENAI_API_KEY"):
                self.llm = ChatOpenAI(model="gpt-4.1")
            else:
                raise ValueError("No LLM provided and no API keys found for Anthropic or OpenAI")
        else:
            self.llm = llm
    
    def _create_context_request(self, document_text: str, chunk_text: str) -> List[Dict[str, Any]]:
        """
        Create a request for generating context for a chunk.
        
        Args:
            document_text: The full document text.
            chunk_text: The text of the chunk to contextualize.
            
        Returns:
            A list of messages for the LLM.
        """
        messages = [
            SystemMessage(content="You are an expert at understanding documents and providing context."),
            HumanMessage(content=self.document_prompt.format(whole_document=document_text) + "\n\n" + 
                            self.chunk_prompt.format(chunk_content=chunk_text))
        ]
        return messages
    
    def contextualize_chunk(self, document_text: str, chunk: Document) -> Document:
        """
        Add contextual information to a single document chunk.
        
        Args:
            document_text: The full document text.
            chunk: The document chunk to contextualize.
            
        Returns:
            A new Document object with contextualized content.
        """
        messages = self._create_context_request(document_text, chunk.page_content)
        context = self.llm.invoke(messages).content
        
        # Create a new document with the context prepended to the original content
        contextualized_content = f"{context}\n\n{chunk.page_content}"
        
        # Create a new Document with the same metadata but updated content
        new_metadata = chunk.metadata.copy()
        new_metadata["original_content"] = chunk.page_content
        new_metadata["context"] = context
        
        return Document(page_content=contextualized_content, metadata=new_metadata)
    
    def contextualize_chunks(self, document_text: str, chunks: List[Document]) -> List[Document]:
        """
        Add contextual information to a list of document chunks.
        
        Args:
            document_text: The full document text.
            chunks: A list of document chunks to contextualize.
            
        Returns:
            A list of Document objects with contextualized content.
        """
        contextualized_chunks = []
        
        for i, chunk in enumerate(chunks):
            print(f"Contextualizing chunk {i+1}/{len(chunks)}...")
            contextualized_chunk = self.contextualize_chunk(document_text, chunk)
            contextualized_chunks.append(contextualized_chunk)
            
        return contextualized_chunks
    
    def contextualize_document(self, document: Document) -> List[Document]:
        """
        Process a document with pre-split chunks (from its metadata).
        
        Args:
            document: A Document object potentially containing chunk information in metadata.
            
        Returns:
            A list of contextualized Document chunks.
        """
        if "chunks" not in document.metadata:
            raise ValueError("Document metadata does not contain pre-split chunks")
            
        chunks = document.metadata["chunks"]
        return self.contextualize_chunks(document.page_content, chunks)
    
    def get_contextual_prompt_with_cache(self, document_text: str, chunk_text: str) -> Dict[str, Any]:
        """
        Create a prompt for contextual retrieval that can utilize Anthropic's prompt caching.
        
        Args:
            document_text: The full document text.
            chunk_text: The text of the chunk to contextualize.
            
        Returns:
            A dictionary with prompt information that can be used with Anthropic's API.
        """
        # Note: This is specifically for Anthropic's API to utilize prompt caching
        prompt = {
            "system": "You are an expert at understanding documents and providing context.",
            "messages": [
                {
                    "role": "user",
                    "content": self.document_prompt.format(whole_document=document_text),
                    "cache_control": {"type": "document"}  # Use caching for the document part
                },
                {
                    "role": "user",
                    "content": self.chunk_prompt.format(chunk_content=chunk_text),
                    "cache_control": {"type": "ephemeral"}  # No caching for the chunk part
                }
            ]
        }
        return prompt


# Utility functions for integrating with the existing RAG system

def contextualize_chunks_for_retrieval(
    document_text: str, 
    chunks: List[Document], 
    llm: Optional[BaseChatModel] = None
) -> List[Document]:
    """
    Utility function to contextualize chunks for retrieval.
    
    Args:
        document_text: The full document text.
        chunks: A list of document chunks to contextualize.
        llm: A language model for generating contextual information.
        
    Returns:
        A list of contextualized Document objects.
    """
    contextualizer = ContextualRetrieval(llm=llm)
    return contextualizer.contextualize_chunks(document_text, chunks)
