from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from LegalResearch import process_question, search_query, retriever, prompt, llm  # Import functions and objects from rag.py

# Define a Pydantic model for request validation
class RequestState(BaseModel):
    messages: List[str]  # List of messages from the user

# Create FastAPI app instance
app = FastAPI(title="Graphrag")

@app.post("/chat")
def chat_endpoint(request: RequestState): 
    """
    API Endpoint to interact with the Chatbot using LangGraph and search tools.
    It dynamically selects the model specified in the request.
    """

    query = request.messages[0]  # Use the first message as the query

    # Call the AI agent function to process the query and get a response
    response = process_question(query, search_query, retriever, prompt, llm)
    
    # Return the AI-generated response
    return {"response": response}


# Step 3: Run app & Explore Swagger UI Docs
if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI application on localhost at port 9999
    uvicorn.run(app, host="127.0.0.1", port=9999)