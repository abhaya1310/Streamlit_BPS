

# RAG-based Policy Retrieval System for Boston Public School

This repository contains a Retrieval-Augmented Generation (RAG) system built to assist the administration of Boston Public Schools by retrieving and generating precise policy answers. Developed as part of my BU practicum, the system leverages state-of-the-art language models, document retrieval, and a custom prompt template to ensure answers remain strictly within the provided policy context.



## Overview

The RAG-based Policy Retrieval System combines document retrieval with language model generation to provide context-specific answers to policy-related questions. The system is designed to:
- Retrieve relevant policy documents from a FAISS vector store.
- Generate precise answers using a custom prompt that limits responses to the given policy context.
- Provide proper citations and reference links for all the documents used to formulate the response.

The project uses Streamlit to create an interactive chat interface for administrators, making it easy to query policy-related information in real time.

## Features

- **Context-aware Responses:** Answers are generated solely based on the provided policy context.
- **Document Retrieval:** Uses FAISS vector store and HuggingFace embeddings for efficient similarity search.
- **Custom Prompt Template:** Guides the language model to produce concise, accurate, and context-bound answers.
- **Reference Citation:** Automatically appends references and source links for the retrieved documents.
- **Streamlit Interface:** Provides a user-friendly chat interface for real-time interaction.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/rag-policy-retrieval.git
   cd rag-policy-retrieval
   ```

2. **Install Dependencies:**

   Ensure you have Python 3.7 or higher installed. Then install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables:**

   Create a `.env` file in the root directory to store your API keys and other environment variables (e.g., OpenAI API key).

   Example `.env` file:

   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

4. **Prepare the FAISS Index:**

   Ensure that the FAISS index (`faiss_index`) is created and available in your repository or local environment. This index should be generated using your corpus of policy documents and the HuggingFace embeddings model.

## Usage

1. **Run the Streamlit App:**

   Launch the app using Streamlit:

   ```bash
   streamlit run app.py
   ```

2. **Interact with the Chat Interface:**

   - Type a policy-related question in the input box.
   - The system retrieves the relevant policy documents and generates an answer using the RAG chain.
   - If the question is out of scope, the system will notify the user and append a marker `[0]__[0]` to indicate the answer was adjusted accordingly.
   - Retrieved document sources and reference links are displayed at the end of the answer.


## Configuration

- **Embeddings Model:**  
  The system uses `sentence-transformers/all-MiniLM-l6-v2` via HuggingFace for generating embeddings.

- **Language Model:**  
  Integrated with OpenAI's model via `ChatOpenAI` (configured with model `"gpt-4o-mini"`).

- **Prompt Template:**  
  A custom template guides the response generation to only use the provided policy context. Ensure that your policy documents are correctly indexed in the FAISS vector store.

## Credits

- Developed as part of a BU practicum.
- Built using [Streamlit](https://streamlit.io/), [Langchain](https://github.com/hwchase17/langchain), and [FAISS](https://github.com/facebookresearch/faiss).
- Special thanks to the Boston Public Schools administration for their input and guidance on policy requirements.
