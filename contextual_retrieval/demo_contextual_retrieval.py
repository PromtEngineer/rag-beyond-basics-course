"""
Demo script for Anthropic's Contextual Retrieval implementation.

This script demonstrates how to use the contextual retrieval approach
with the RAG Beyond Basics repository.
"""

import argparse
import os
import sys
from rich import print
from rich.console import Console
from rich.markdown import Markdown

from contextual_rag_client import ContextualRagClient
from rag_client import rag_client

console = Console()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demo for Contextual Retrieval RAG")
    parser.add_argument('--file', type=str, required=True, help='Path to the PDF file')
    parser.add_argument('--compare', action='store_true', help='Compare with standard RAG')
    parser.add_argument('--anthropic', action='store_true', help='Use Anthropic Claude (requires API key)')
    return parser.parse_args()

def check_environment():
    """Check if the required environment variables are set."""
    required_vars = []
    
    if not os.environ.get("OPENAI_API_KEY"):
        required_vars.append("OPENAI_API_KEY")
    
    if args.anthropic and not os.environ.get("ANTHROPIC_API_KEY"):
        required_vars.append("ANTHROPIC_API_KEY")
    
    if os.environ.get("COHERE_API_KEY") is None:
        print("[yellow]Warning: COHERE_API_KEY is not set. Reranking will default to GPT.[/yellow]")
    
    if required_vars:
        print(f"[red]Error: The following environment variables are required but not set: {', '.join(required_vars)}[/red]")
        sys.exit(1)

def print_header(text):
    """Print a formatted header."""
    console.print(f"\n[bold blue]{'=' * 20} {text} {'=' * 20}[/bold blue]\n")

def main(args):
    """Run the demo."""
    check_environment()
    
    print_header("Initializing RAG Systems")
    
    # Initialize the contextual RAG client
    console.print("[bold green]Initializing Contextual RAG system...[/bold green]")
    contextual_client = ContextualRagClient(
        files=args.file,
        use_anthropic=args.anthropic
    )
    
    # Initialize the standard RAG client if comparison is requested
    standard_client = None
    if args.compare:
        console.print("[bold green]Initializing Standard RAG system for comparison...[/bold green]")
        standard_client = rag_client(files=args.file)
    
    # Interactive query loop
    while True:
        print_header("Interactive Q&A")
        query = console.input("[bold yellow]Enter your question (or 'exit' to quit): [/bold yellow]")
        
        if query.lower() == 'exit':
            break
        
        # Generate response with contextual RAG
        print_header("Contextual RAG Response")
        console.print("[bold]Query:[/bold] " + query)
        response = contextual_client.generate(query)
        console.print("\n[bold]Response:[/bold]")
        console.print(Markdown(response["response"]))
        
        # Generate response with standard RAG if comparison is requested
        if args.compare and standard_client:
            print_header("Standard RAG Response")
            std_response = standard_client.generate(query)
            console.print("\n[bold]Response:[/bold]")
            console.print(Markdown(std_response["response"]))
        
        # Ask if user wants to see the retrieved contexts
        show_context = console.input("\n[bold yellow]Show retrieved contexts? (y/n): [/bold yellow]").lower() == 'y'
        if show_context:
            print_header("Retrieved Contexts")
            console.print(response["contexts"])

if __name__ == "__main__":
    args = parse_args()
    main(args)
