from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import torch
import time
import json
import cohere
import os


# Function to compute MaxSim
def maxsim(query_embedding, document_embedding):
    # Expand dimensions for broadcasting
    # Query: [batch_size, query_length, embedding_size] -> [batch_size, query_length, 1, embedding_size]
    # Document: [batch_size, doc_length, embedding_size] -> [batch_size, 1, doc_length, embedding_size]
    expanded_query = query_embedding.unsqueeze(2)
    expanded_doc = document_embedding.unsqueeze(1)

    # Compute cosine similarity across the embedding dimension
    sim_matrix = torch.nn.functional.cosine_similarity(expanded_query, expanded_doc, dim=-1)

    # Take the maximum similarity for each query token (across all document tokens)
    # sim_matrix shape: [batch_size, query_length, doc_length]
    max_sim_scores, _ = torch.max(sim_matrix, dim=2)

    # Average these maximum scores across all query tokens
    avg_max_sim = torch.mean(max_sim_scores, dim=1)
    return avg_max_sim

def reranking_gpt(similar_chunks, query):

    start = time.time()
    client = OpenAI()
    response = client.chat.completions.create(
        model='gpt-4-1106-preview',
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
        {"role": "system", 
        "content": """You are an expert relevance ranker. Given a list of documents and a query, your job is to determine how relevant each document is for answering the query. 
        Your output is JSON, which is a list of documents.  Each document has two fields, content and score.  relevance_score is from 0.0 to 100.0. Higher relevance means higher score."""},
        {"role": "user", "content": f"Query: {query} Docs: {similar_chunks}"}
        ]
    )

    print(f"Took {time.time() - start} seconds to re-rank documents with GPT-4.")

    # Sort the scores by highest to lowest and print
    scores = json.loads(response.choices[0].message.content)["documents"]
    sorted_data = sorted(scores, key=lambda x: x['score'], reverse=True)

    documents = []
    for idx, r in enumerate(sorted_data):
        documents.append(f"{r['content']}")
    return documents


def reranking_colbert(similar_chunks, query):

    start = time.time()
    scores = []

    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
    model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")

    # Encode the query
    query_encoding = tokenizer(query, return_tensors='pt')
    query_embedding = model(**query_encoding).last_hidden_state.mean(dim=1)

    # Get score for each document
    for document in similar_chunks:
        document_encoding = tokenizer(document.page_content, return_tensors='pt', truncation=True, max_length=512)
        document_embedding = model(**document_encoding).last_hidden_state

        # Calculate MaxSim score
        score = maxsim(query_embedding.unsqueeze(0), document_embedding)
        scores.append({
            "score": score.item(),
            "document": document.page_content,
        })

    print(f"Took {time.time() - start} seconds to re-rank documents with ColBERT.")

    # Sort the scores by highest to lowest and print
    sorted_data = sorted(scores, key=lambda x: x['score'], reverse=True)
    documents = []
    for idx, r in enumerate(sorted_data):
        documents.append(f"{r['document']}")
    
    return documents


def reranking_cohere(similar_chunks, query):
    # Get your cohere API key on: www.cohere.com
    co = cohere.Client(os.environ["COHERE_API_KEY"])

    documents = [f"{doc.page_content}" for doc in similar_chunks]
    # Example query and passages
    start = time.time()

    results = co.rerank(query=query, 
                        documents=documents, 
                        top_n=4, 
                        model="rerank-english-v3.0", 
                        return_documents=True)
    print(f"Took {time.time() - start} seconds to re-rank documents with Cohere.")

    documents = []
    for idx, r in enumerate(results.results):
        documents.append(f"{r.document.text}")

    return documents

