
# RAG Beyond Basics

Welcome to the [RAG Beyond Basics course](https://prompt-s-site.thinkific.com/courses/rag). This repository contains the implementation of a Retrieval-Augmented Generation (RAG) system with query expansion, context expansion, and reranking. The instructions below will guide you through setting up and running the project.

## Enroll in the Course

This repo is complimentary to the RAG Beyond Basics Course, [Enroll](https://prompt-s-site.thinkific.com/courses/rag) here. 

## Table of Contents

- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

- `app_streamlit.py`: Streamlit application script
- `app.py`: Main application script
- `RAG_Beyond_Basics.ipynb`: Jupyter notebook for the course
- `rag_client.py`: Client script for RAG system
- `rag.py`: Core RAG system implementation
- `reranking_models.py`: Models for reranking the results
- `requirements.txt`: List of required packages

## Setup Instructions

### Prerequisites

Before you begin, ensure you have the following installed on your machine:

- Python 3.10 or later
- Conda
- Git

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/PromtEngineer/rag-beyond-basics-course.git
   cd rag-beyond-basics-course
   ```

2. **Create a virtual environment**

   - **Linux and macOS**

     ```bash
     conda create --name rag-beyond-basics python=3.10
     conda activate rag-beyond-basics
     ```

   - **Windows**

     ```bash
     conda create --name rag-beyond-basics python=3.10
     conda activate rag-beyond-basics
     ```

3. **Install the required packages**

   ```bash
   pip install -r requirements.txt
   ```

### Environment Variables

Set the following environment variables with your respective API keys:

- `OPENAI_API_KEY`: Your OpenAI API key
- `CO_API_KEY`: Your Cohere API key

You can set these environment variables in your terminal:

- **Linux and macOS**

  ```bash
  export OPENAI_API_KEY='your-openai-api-key'
  export CO_API_KEY='your-cohere-api-key'
  ```

- **Windows**

  ```bash
  set OPENAI_API_KEY=your-openai-api-key
  set CO_API_KEY=your-cohere-api-key
  ```

## Running the Application

To run the Streamlit application, execute the following command:

```bash
streamlit run app_streamlit.py
```

This will start the Streamlit server, and you can interact with the application through your web browser.

## Usage

1. Open your web browser and navigate to the URL provided by the Streamlit server (usually `http://localhost:8501`).
2. Follow the on-screen instructions to use the RAG system with query expansion, context expansion, and reranking.



