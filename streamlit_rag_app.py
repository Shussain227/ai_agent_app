import streamlit as st
import openai
import chromadb
import json
import uuid
from datetime import datetime
import tiktoken
from typing import Dict, Any, List
import io
import tempfile
import os
import traceback

# Page config
st.set_page_config(
    page_title="AI Agents Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'simple_agent' not in st.session_state:
    st.session_state.simple_agent = None
if 'rag_agent' not in st.session_state:
    st.session_state.rag_agent = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_chat_history' not in st.session_state:
    st.session_state.rag_chat_history = []

class SimpleTaskAgent:
    """Simple Task Agent with weather and calculator tools"""
    
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather information for a specific location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name or location"
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5 + 3')"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]
    
    def get_weather(self, location: str) -> Dict[str, Any]:
        """Get weather information for a location (mock data)"""
        mock_weather = {
            "london": {"temp": 15, "desc": "cloudy", "humidity": 80},
            "tokyo": {"temp": 22, "desc": "sunny", "humidity": 65},
            "new york": {"temp": 18, "desc": "partly cloudy", "humidity": 70},
            "paris": {"temp": 12, "desc": "rainy", "humidity": 85},
            "mumbai": {"temp": 28, "desc": "hot and humid", "humidity": 85},
            "sydney": {"temp": 20, "desc": "mild", "humidity": 60}
        }
        
        location_lower = location.lower()
        if location_lower in mock_weather:
            data = mock_weather[location_lower]
            return {
                "location": location,
                "temperature": data["temp"],
                "description": data["desc"],
                "humidity": data["humidity"],
                "feels_like": data["temp"] - 2
            }
        else:
            return {
                "location": location,
                "temperature": 20,
                "description": "clear sky",
                "humidity": 60,
                "feels_like": 18
            }
    
    def calculate(self, expression: str) -> Dict[str, Any]:
        """Perform mathematical calculations safely"""
        try:
            allowed_chars = set('0123456789+-*/().^ ')
            if not all(c in allowed_chars for c in expression.replace(' ', '')):
                return {"error": "Invalid characters in expression"}
            
            expression = expression.replace('^', '**')
            result = eval(expression)
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"error": f"Calculation error: {str(e)}"}
    
    def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a function call"""
        if function_name == "get_weather":
            return self.get_weather(arguments["location"])
        elif function_name == "calculate":
            return self.calculate(arguments["expression"])
        else:
            return {"error": f"Unknown function: {function_name}"}
    
    def chat(self, user_message: str) -> str:
        """Main chat interface with tool calling"""
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that can check weather and perform calculations.
                
When users ask about weather, use the get_weather function.
When users ask for calculations, use the calculate function.
Always be helpful and provide clear, conversational responses."""
            },
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message
            
            if assistant_message.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": assistant_message.tool_calls
                })
                
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    function_result = self.execute_function(function_name, function_args)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(function_result)
                    })
                
                final_response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=messages
                )
                return final_response.choices[0].message.content
            else:
                return assistant_message.content
                
        except Exception as e:
            return f"Error: {str(e)}"

class RAGAgent:
    """RAG-Enhanced Agent with document knowledge and memory"""
    
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Initialize ChromaDB with persistent storage
        try:
            # Use a temporary directory for ChromaDB
            self.chroma_client = chromadb.Client()
            
            # Create collections
            try:
                self.documents_collection = self.chroma_client.create_collection(
                    name=f"documents_{uuid.uuid4().hex[:8]}",
                    metadata={"description": "Document knowledge base"}
                )
            except:
                self.documents_collection = self.chroma_client.get_collection("documents")
                
            try:
                self.memory_collection = self.chroma_client.create_collection(
                    name=f"memory_{uuid.uuid4().hex[:8]}",
                    metadata={"description": "Conversation memory"}
                )
            except:
                self.memory_collection = self.chroma_client.get_collection("memory")
        except Exception as e:
            st.error(f"Error initializing ChromaDB: {e}")
            raise
        
        self.conversation_history = []
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Search through uploaded documents for relevant information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find relevant document content"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 3)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "recall_conversation",
                    "description": "Recall previous conversation context related to a topic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Topic or keyword to search for in conversation history"
                            }
                        },
                        "required": ["topic"]
                    }
                }
            }
        ]
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI's embedding model"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error getting embedding: {e}")
            return [0.0] * 1536
    
    def split_text(self, text: str, max_tokens: int = 500) -> List[str]:
        """Split text into chunks of specified token length"""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    def add_document(self, content: str, title: str = None, metadata: Dict = None):
        """Add a document to the knowledge base"""
        doc_id = str(uuid.uuid4())
        chunks = self.split_text(content, max_tokens=500)
        
        successful_chunks = 0
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_metadata = {
                "title": title or "Untitled Document",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            try:
                embedding = self.get_embedding(chunk)
                self.documents_collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[chunk_metadata],
                    ids=[chunk_id]
                )
                successful_chunks += 1
            except Exception as e:
                st.error(f"Error adding document chunk: {e}")
        
        return successful_chunks, len(chunks)
    
    def search_documents(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search documents for relevant information"""
        try:
            query_embedding = self.get_embedding(query)
            
            results = self.documents_collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results
            )
            
            formatted_results = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0
                    
                    formatted_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "relevance_score": 1 - distance
                    })
            
            return formatted_results
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]
    
    def recall_conversation(self, topic: str) -> List[Dict[str, Any]]:
        """Recall relevant conversation history"""
        try:
            topic_embedding = self.get_embedding(topic)
            
            results = self.memory_collection.query(
                query_embeddings=[topic_embedding],
                n_results=3
            )
            
            recalled_items = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    
                    recalled_items.append({
                        "content": doc,
                        "timestamp": metadata.get("timestamp", ""),
                        "context": metadata.get("context", "")
                    })
            
            return recalled_items
        except Exception as e:
            return [{"error": f"Recall failed: {str(e)}"}]
    
    def save_to_memory(self, user_message: str, assistant_response: str):
        """Save conversation exchange to memory"""
        try:
            memory_id = str(uuid.uuid4())
            memory_content = f"User: {user_message}\nAssistant: {assistant_response}"
            
            embedding = self.get_embedding(memory_content)
            
            self.memory_collection.add(
                documents=[memory_content],
                embeddings=[embedding],
                metadatas=[{
                    "timestamp": datetime.now().isoformat(),
                    "context": "conversation",
                    "user_message": user_message,
                    "assistant_response": assistant_response
                }],
                ids=[memory_id]
            )
        except Exception as e:
            st.error(f"Error saving to memory: {e}")
    
    def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a function call"""
        if function_name == "search_documents":
            query = arguments["query"]
            max_results = arguments.get("max_results", 3)
            return {"results": self.search_documents(query, max_results)}
        elif function_name == "recall_conversation":
            topic = arguments["topic"]
            return {"recalled_items": self.recall_conversation(topic)}
        else:
            return {"error": f"Unknown function: {function_name}"}
    
    def chat(self, user_message: str) -> str:
        """Main chat interface with RAG and memory"""
        self.conversation_history.append({"role": "user", "content": user_message})
        
        system_message = """You are a helpful AI assistant with access to a document knowledge base and conversation memory.

When users ask questions:
1. Use search_documents to find relevant information from uploaded documents
2. Use recall_conversation to remember previous discussions on related topics
3. Combine information from documents, memory, and your general knowledge to provide comprehensive answers
4. Always cite when you're using information from documents or previous conversations

Be conversational and helpful while being accurate and well-informed."""

        messages = [
            {"role": "system", "content": system_message},
            *self.conversation_history[-10:],
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message
            
            if assistant_message.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": assistant_message.tool_calls
                })
                
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    function_result = self.execute_function(function_name, function_args)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(function_result)
                    })
                
                final_response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=messages
                )
                final_content = final_response.choices[0].message.content
            else:
                final_content = assistant_message.content
            
            self.conversation_history.append({"role": "assistant", "content": final_content})
            self.save_to_memory(user_message, final_content)
            
            return final_content
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            self.conversation_history.append({"role": "assistant", "content": error_message})
            return error_message

def main():
    st.markdown('<h1 class="main-header">🤖 AI Agents Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "Choose Demo:",
            ["🏠 Home", "🛠️ Simple Task Agent", "📚 RAG-Enhanced Agent"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("## About")
        st.markdown("""
        This demo showcases two types of AI agents:
        
        **Simple Task Agent:**
        - Weather information
        - Mathematical calculations
        - Basic tool use
        
        **RAG-Enhanced Agent:**
        - Document knowledge base
        - Conversation memory
        - Advanced information synthesis
        """)
    
    # API Key input
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("## Configuration")
        api_key = st.text_input(
            "OpenAI API Key:",
            type="password",
            value=st.session_state.openai_api_key,
            help="Enter your OpenAI API key to use the agents"
        )
        
        if api_key != st.session_state.openai_api_key:
            st.session_state.openai_api_key = api_key
            # Reset agents when API key changes
            st.session_state.simple_agent = None
            st.session_state.rag_agent = None
    
    if not st.session_state.openai_api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to use the agents.")
        st.info("💡 You can get an API key from: https://platform.openai.com/api-keys")
        return
    
    # Home Page
    if page == "🏠 Home":
        show_home_page()
    
    # Simple Task Agent
    elif page == "🛠️ Simple Task Agent":
        show_simple_agent_page()
    
    # RAG-Enhanced Agent
    elif page == "📚 RAG-Enhanced Agent":
        show_rag_agent_page()

def show_home_page():
    st.markdown("## Welcome to the AI Agents Demo!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>🛠️ Simple Task Agent</h3>
            <p>Demonstrates basic AI agent capabilities:</p>
            <ul>
                <li>Weather information retrieval</li>
                <li>Mathematical calculations</li>
                <li>Tool use and function calling</li>
                <li>Conversational interface</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>📚 RAG-Enhanced Agent</h3>
            <p>Advanced agent with knowledge capabilities:</p>
            <ul>
                <li>Document upload and processing</li>
                <li>Semantic search through documents</li>
                <li>Conversation memory and recall</li>
                <li>Multi-source information synthesis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🚀 Getting Started")
    st.markdown("""
    1. **Enter your OpenAI API Key** in the sidebar
    2. **Choose an agent** from the navigation menu
    3. **Start chatting** and explore the capabilities!
    
    Both agents demonstrate different aspects of AI agent architecture and can be used to understand:
    - Function calling and tool use
    - Vector databases and RAG (Retrieval Augmented Generation)
    - Conversation memory and context management
    - Multi-modal information processing
    """)

def show_simple_agent_page():
    st.markdown('<h2 class="sub-header">🛠️ Simple Task Agent</h2>', unsafe_allow_html=True)
    
    # Initialize agent
    if st.session_state.simple_agent is None:
        try:
            with st.spinner("Initializing Simple Task Agent..."):
                st.session_state.simple_agent = SimpleTaskAgent(st.session_state.openai_api_key)
            st.success("✅ Simple Task Agent initialized successfully!")
        except Exception as e:
            st.error(f"❌ Error initializing agent: {str(e)}")
            return
    
    # Agent capabilities
    st.markdown("""
    <div class="info-box">
        <h4>Agent Capabilities:</h4>
        <ul>
            <li>🌤️ <strong>Weather Information:</strong> Ask about weather in any city</li>
            <li>🧮 <strong>Mathematical Calculations:</strong> Perform complex calculations</li>
            <li>💬 <strong>Natural Conversation:</strong> Conversational interface with tool integration</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Example prompts
    st.markdown("### 💡 Try these examples:")
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        if st.button("☀️ Weather in Tokyo", key="weather_example"):
            st.session_state.current_query = "What's the weather like in Tokyo?"
    
    with example_col2:
        if st.button("🧮 Calculate 15 * 8 + 32", key="calc_example"):
            st.session_state.current_query = "Can you calculate 15 * 8 + 32?"
    
    with example_col3:
        if st.button("🌍 Weather + Math", key="multi_example"):
            st.session_state.current_query = "What's the weather in London and also calculate the square root of 144?"
    
    # Chat interface
    st.markdown("---")
    st.markdown("### 💬 Chat with Simple Task Agent")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Agent:** {message['content']}")
    
    # Chat input
    query = st.text_input(
        "Ask the agent something:",
        value=getattr(st.session_state, 'current_query', ''),
        key="simple_chat_input"
    )
    
    if st.button("Send", key="simple_send") and query:
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.simple_agent.chat(query)
                
                # Add to chat history
                st.session_state.chat_history.append({"role": "user", "content": query})
                st.session_state.chat_history.append({"role": "agent", "content": response})
                
                # Clear current query
                if hasattr(st.session_state, 'current_query'):
                    delattr(st.session_state, 'current_query')
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Clear chat history
    if st.button("Clear Chat History", key="clear_simple_chat"):
        st.session_state.chat_history = []
        st.rerun()

def show_rag_agent_page():
    st.markdown('<h2 class="sub-header">📚 RAG-Enhanced Agent</h2>', unsafe_allow_html=True)
    
    # Initialize agent
    if st.session_state.rag_agent is None:
        try:
            with st.spinner("Initializing RAG Agent with ChromaDB..."):
                st.session_state.rag_agent = RAGAgent(st.session_state.openai_api_key)
            st.success("✅ RAG Agent initialized successfully!")
        except Exception as e:
            st.error(f"❌ Error initializing RAG agent: {str(e)}")
            st.error("Make sure ChromaDB is properly installed: `pip install chromadb`")
            return
    
    # Agent capabilities
    st.markdown("""
    <div class="info-box">
        <h4>RAG Agent Capabilities:</h4>
        <ul>
            <li>📄 <strong>Document Processing:</strong> Upload and process text documents</li>
            <li>🔍 <strong>Semantic Search:</strong> Find relevant information using vector similarity</li>
            <li>🧠 <strong>Memory Management:</strong> Remember and recall conversation context</li>
            <li>🔄 <strong>Multi-source Synthesis:</strong> Combine information from multiple sources</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Document upload section
    st.markdown("### 📁 Document Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload a text document (.txt, .md)",
            type=['txt', 'md'],
            help="Upload documents to add to the agent's knowledge base"
        )
    
    with col2:
        if st.button("📋 Add Sample Documents", key="add_sample_docs"):
            # Add sample documents
            sample_docs = [
                {
                    "title": "Company Policy Manual",
                    "content": """
                    Company Remote Work Policy:
                    
                    All employees are eligible for remote work arrangements. Employees must:
                    1. Maintain regular working hours (9 AM - 5 PM local time)
                    2. Attend mandatory team meetings via video call
                    3. Submit weekly progress reports every Friday
                    4. Ensure reliable internet connection and dedicated workspace
                    
                    Vacation Policy:
                    Employees accrue 2.5 vacation days per month, up to 30 days annually.
                    Vacation requests must be submitted 2 weeks in advance through the HR portal.
                    Emergency leave can be requested with manager approval.
                    """
                },
                {
                    "title": "Technical Documentation",
                    "content": """
                    API Authentication Guide:
                    
                    Our REST API uses JWT (JSON Web Token) authentication. To authenticate:
                    1. Send POST request to /auth/login with username and password
                    2. Receive JWT token in the response body
                    3. Include token in Authorization header: Bearer <token>
                    4. Tokens expire after 24 hours and must be refreshed
                    
                    Rate Limiting:
                    API calls are limited to 1000 requests per hour per user.
                    Exceeded limits result in HTTP 429 status code.
                    Premium accounts get 5000 requests per hour.
                    """
                }
            ]
            
            with st.spinner("Adding sample documents..."):
                for doc in sample_docs:
                    success, total = st.session_state.rag_agent.add_document(
                        doc["content"], 
                        doc["title"], 
                        {"category": "sample"}
                    )
                    
            st.success("✅ Sample documents added to knowledge base!")
    
    # Handle file upload
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode('utf-8')
            filename = uploaded_file.name
            
            with st.spinner(f"Processing {filename}..."):
                success, total = st.session_state.rag_agent.add_document(
                    content, 
                    filename, 
                    {"source": "upload"}
                )
            
            st.success(f"✅ Document processed: {success}/{total} chunks added to knowledge base")
            
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
    
    # Example prompts for RAG
    st.markdown("### 💡 Try these examples:")
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        if st.button("📋 Company Policy", key="policy_example"):
            st.session_state.current_rag_query = "What is the company's remote work policy?"
    
    with example_col2:
        if st.button("🔐 API Authentication", key="api_example"):
            st.session_state.current_rag_query = "How do I authenticate with the API?"
    
    with example_col3:
        if st.button("🧠 Recall Previous", key="recall_example"):
            st.session_state.current_rag_query = "What did we discuss about vacation policies?"
    
    # Chat interface
    st.markdown("---")
    st.markdown("### 💬 Chat with RAG Agent")
    
    # Display chat history
    for message in st.session_state.rag_chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**RAG Agent:** {message['content']}")
    
    # Chat input
    rag_query = st.text_input(
        "Ask about your documents or previous conversations:",
        value=getattr(st.session_state, 'current_rag_query', ''),
        key="rag_chat_input"
    )
    
    if st.button("Send", key="rag_send") and rag_query:
        with st.spinner("Searching documents and generating response..."):
            try:
                response = st.session_state.rag_agent.chat(rag_query)
                
                # Add to chat history
                st.session_state.rag_chat_history.append({"role": "user", "content": rag_query})
                st.session_state.rag_chat_history.append({"role": "agent", "content": response})
                
                # Clear current query
                if hasattr(st.session_state, 'current_rag_query'):
                    delattr(st.session_state, 'current_rag_query')
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error(traceback.format_exc())
    
    # Clear chat history
    if st.button("Clear Chat History", key="clear_rag_chat"):
        st.session_state.rag_chat_history = []
        st.rerun()
    
    # Agent status
    if st.button("🔍 Show Agent Status", key="agent_status"):
        try:
            doc_count = st.session_state.rag_agent.documents_collection.count()
            memory_count = st.session_state.rag_agent.memory_collection.count()
            
            st.markdown(f"""
            <div class="success-box">
                <h4>📊 Agent Status:</h4>
                <ul>
                    <li>📄 <strong>Document chunks:</strong> {doc_count}</li>
                    <li>🧠 <strong>Memory entries:</strong> {memory_count}</li>
                    <li>💬 <strong>Current conversation length:</strong> {len(st.session_state.rag_agent.conversation_history)}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error getting status: {e}")

if __name__ == "__main__":
    main()