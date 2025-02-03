from typing import List,Tuple
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
import os

# Load environment variables
env_vars = {
    "NEO4J_URI": os.environ.get("NEO4J_URI"),
    "NEO4J_USERNAME": os.environ.get("NEO4J_USERNAME"),
    "NEO4J_PASSWORD": os.environ.get("NEO4J_PASSWORD"),
    "GROQ_API_KEY": os.environ.get("GROQ_API_KEY"),
    "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY")
}

# Set environment variables
os.environ.update(env_vars)

# Neo4j graph database connection
graph = Neo4jGraph(url=env_vars["NEO4J_URI"], username=env_vars["NEO4J_USERNAME"], password=env_vars["NEO4J_PASSWORD"])

# Google Generative AI Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=env_vars["GEMINI_API_KEY"])

# Vector store using Neo4j
vector_index = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

# ChatGROQ setup
llm = ChatGroq(temperature=0, model="gemma2-9b-it")

# Define a model to extract entities from text
class Entities(BaseModel):
    names: List[str] = Field(..., description="List of person, organization, or business entities found in the text.")

def create_entity_extraction_prompt() -> ChatPromptTemplate:
    """Creates a chat prompt for entity extraction."""
    return ChatPromptTemplate.from_messages([
        ("system", "You are extracting organization and person entities from the text."),
        ("human", "Use the given format to extract information from the following input: {question}"),
    ])

def build_entity_extraction_chain(llm) -> object:
    """Builds the entity extraction chain using the LLM model."""
    return create_entity_extraction_prompt() | llm.with_structured_output(Entities)

entity_chain = build_entity_extraction_chain(llm)

def generate_full_text_query(input_text: str) -> str:
    """Generates a full-text index query for Neo4j."""
    words = [word for word in remove_lucene_chars(input_text).split() if word]
    return " AND ".join(f"{word}~2" for word in words)

def structured_retriever(question: str) -> str:
    """Retrieves structured data from the Neo4j graph based on extracted entities."""
    entities = entity_chain.invoke({"question": question})
    result = []

    for entity in entities.names:
        query = generate_full_text_query(entity)
        response = graph.query(
            """
            CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node, score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT 50
            """, {"query": query}
        )
        result.extend(el['output'] for el in response)

    return "\n".join(result)

def retriever(question: str) -> str:
    """Combines structured and unstructured data retrieval."""
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    
    return f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ".join(unstructured_data)}
"""

# Define the standalone question rephrasing prompt
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow-up question, 
    rephrase the follow-up question to be a standalone question in its original language.
Chat History:
{chat_history}
Follow-Up Input: {question}
Standalone Question:"""
)

def format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    """Formats chat history into a structured message format."""
    return [msg for human, ai in chat_history for msg in (HumanMessage(content=human), AIMessage(content=ai))]

# Define a runnable for handling search queries
search_query = RunnableBranch(
    # If chat history exists, condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
        RunnablePassthrough.assign(chat_history=lambda x: format_chat_history(x["chat_history"])) | CONDENSE_QUESTION_PROMPT | ChatGroq(temperature=0) | StrOutputParser(),
    ),
    # Otherwise, pass through the question directly
    RunnableLambda(lambda x: x["question"]),
)

# Define a prompt template for answering questions based on context
template = """Answer the question based only on the following context:
{context}
Question: {question}
Use natural language and be concise.
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

def process_question(question, search_query, retriever, prompt, llm):
    """Process the question through a defined pipeline."""
    chain = (
        RunnableParallel(
            {"context": search_query | retriever, "question": RunnablePassthrough()},
        )
        | prompt | llm | StrOutputParser()
    )
    
    return chain.invoke({"question": question})