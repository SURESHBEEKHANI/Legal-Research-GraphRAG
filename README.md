# Legal Research Assistant

This project is an AI-powered legal research assistant that leverages LangChain, Neo4j, and Google Generative AI to provide comprehensive legal analysis, case law research, statute and regulation lookup, and document drafting assistance.

## Features

- ⚖️ Comprehensive Legal Analysis
- 🔍 Case Law Research
- 📑 Statute and Regulation Lookup
- 📝 Document Drafting Assistance

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Neo4j database
- Google Generative AI API key
- GROQ API key

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/SURESHBEEKHANI/Legal-Research-GraphRAG.git
    cd Legal-Research-GraphRAG
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:
    ```bash
    export NEO4J_URI=your_neo4j_uri
    export NEO4J_USERNAME=your_neo4j_username
    export NEO4J_PASSWORD=your_neo4j_password
    export GROQ_API_KEY=your_groq_api_key
    export GEMINI_API_KEY=your_gemini_api_key
    ```

### Running the Application

1. Start the FastAPI backend:
    ```bash
    uvicorn backend:app --host 127.0.0.1 --port 9999
    ```

2. Start the Streamlit frontend:
    ```bash
    streamlit run app.py
    ```

### Usage

1. Open your web browser and navigate to `http://localhost:8501`.
2. Enter your legal query in the input box and interact with the AI-powered assistant.

## Project Structure

- `LegalResearch.py`: Contains the core logic for entity extraction, data retrieval, and question processing.
- `app.py`: Streamlit application for the frontend interface.
- `backend.py`: FastAPI application for the backend API.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.