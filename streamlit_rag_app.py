import streamlit as st
import sys
import traceback

# Page config
st.set_page_config(
    page_title="AI Agents Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import required packages with error handling
try:
    import openai
    import json
    import uuid
    from datetime import datetime
    from typing import Dict, Any, List
    OPENAI_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Error importing OpenAI: {e}")
    OPENAI_AVAILABLE = False

try:
    import chromadb
    import tiktoken
    CHROMADB_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Error importing ChromaDB/Tiktoken: {e}")
    CHROMADB_AVAILABLE = False

# Custom CSS
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
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available")
            
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
                                "description": "Mathematical expression to evaluate"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]
    
    def get_weather(self, location: str) -> Dict[str, Any]:
        """Get weather information (mock data for demo)"""
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
                "content": "You are a helpful assistant that can check weather and perform calculations. Use the available functions when appropriate."
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
        if not OPENAI_AVAILABLE or not CHROMADB_AVAILABLE:
            raise ImportError("Required packages not available")
            
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Initialize ChromaDB
        try:
            self.chroma_client = chromadb.Client()
            
            # Create unique collection names to avoid conflicts
            doc_name = f"docs_{uuid.uuid4().hex[:8]}"
            mem_name = f"memory_{uuid.uuid4().hex[:8]}"
            
            self.documents_collection = self.chroma_client.create_collection(
                name=doc_name,
                metadata={"description": "Document knowledge base"}
            )
            
            self.memory_collection = self.chroma_client.create_collection(
                name=mem_name,
                metadata={"description": "Conversation memory"}
            )
            
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
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "description": "Max results", "default": 3}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
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
        """Split text into chunks"""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    def add_document(self, content: str, title: str = None):
        """Add a document to the knowledge base"""
        doc_id = str(uuid.uuid4())
        chunks = self.split_text(content, max_tokens=500)
        
        successful_chunks = 0
        
        for i, chunk in enumerate(chunks):
            try:
                embedding = self.get_embedding(chunk)
                self.documents_collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{
                        "title": title or "Document",
                        "chunk_index": i,
                        "timestamp": datetime.now().isoformat(),
                    }],
                    ids=[f"{doc_id}_chunk_{i}"]
                )
                successful_chunks += 1
            except Exception as e:
                st.error(f"Error adding chunk {i}: {e}")
        
        return successful_chunks, len(chunks)
    
    def search_documents(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search documents"""
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
                    formatted_results.append({
                        "content": doc,
                        "metadata": metadata
                    })
            
            return formatted_results
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]
    
    def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute function call"""
        if function_name == "search_documents":
            query = arguments["query"]
            max_results = arguments.get("max_results", 3)
            return {"results": self.search_documents(query, max_results)}
        else:
            return {"error": f"Unknown function: {function_name}"}
    
    def chat(self, user_message: str) -> str:
        """Chat with RAG capabilities"""
        self.conversation_history.append({"role": "user", "content": user_message})
        
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful AI assistant with access to documents. Use search_documents to find relevant information when needed."
            },
            *self.conversation_history[-10:]
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
            return final_content
            
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    st.markdown('<h1 class="main-header">🤖 AI Agents Demo</h1>', unsafe_allow_html=True)
    
    # Check system status
    st.sidebar.markdown("## 📊 System Status")
    
    if OPENAI_AVAILABLE:
        st.sidebar.success("✅ OpenAI Available")
    else:
        st.sidebar.error("❌ OpenAI Not Available")
    
    if CHROMADB_AVAILABLE:
        st.sidebar.success("✅ ChromaDB Available") 
    else:
        st.sidebar.error("❌ ChromaDB Not Available")
    
    # Show Python and package info
    with st.sidebar.expander("🔍 Debug Info"):
        st.text(f"Python: {sys.version}")
        try:
            st.text(f"OpenAI: {openai.__version__}")
        except:
            st.text("OpenAI: Not installed")
        try:
            st.text(f"ChromaDB: {chromadb.__version__}")
        except:
            st.text("ChromaDB: Not installed")
    
    # Navigation
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Choose Demo:", ["🏠 Home", "🛠️ Simple Agent", "📚 RAG Agent"])
    
    # API Key
    st.sidebar.markdown("---")
    api_key = st.sidebar.text_input("OpenAI API Key:", type="password")
    
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar.")
        return
    
    # Pages
    if page == "🏠 Home":
        show_home_page()
    elif page == "🛠️ Simple Agent":
        if OPENAI_AVAILABLE:
            show_simple_agent_page(api_key)
        else:
            st.error("❌ OpenAI package not available. Cannot run Simple Agent.")
    elif page == "📚 RAG Agent":
        if OPENAI_AVAILABLE and CHROMADB_AVAILABLE:
            show_rag_agent_page(api_key)
        else:
            st.error("❌ Required packages not available. Cannot run RAG Agent.")

def show_home_page():
    st.markdown("## Welcome to the AI Agents Demo!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>🛠️ Simple Task Agent</h3>
            <p>Basic AI agent capabilities:</p>
            <ul>
                <li>Weather information</li>
                <li>Mathematical calculations</li>
                <li>Tool use demonstration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>📚 RAG-Enhanced Agent</h3>
            <p>Advanced knowledge capabilities:</p>
            <ul>
                <li>Document upload and processing</li>
                <li>Semantic search</li>
                <li>Information synthesis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def show_simple_agent_page(api_key):
    st.markdown('<h2 class="sub-header">🛠️ Simple Task Agent</h2>', unsafe_allow_html=True)
    
    if st.session_state.simple_agent is None:
        try:
            with st.spinner("Initializing Simple Task Agent..."):
                st.session_state.simple_agent = SimpleTaskAgent(api_key)
            st.success("✅ Agent initialized!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return
    
    # Example buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Weather in Tokyo"):
            st.session_state.current_query = "What's the weather like in Tokyo?"
    with col2:
        if st.button("Calculate 15 * 8 + 32"):
            st.session_state.current_query = "Calculate 15 * 8 + 32"
    with col3:
        if st.button("Weather + Math"):
            st.session_state.current_query = "Weather in London and calculate sqrt(144)"
    
    # Chat
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Agent:** {message['content']}")
    
    query = st.text_input("Ask something:", value=getattr(st.session_state, 'current_query', ''))
    
    if st.button("Send") and query:
        with st.spinner("Processing..."):
            try:
                response = st.session_state.simple_agent.chat(query)
                st.session_state.chat_history.append({"role": "user", "content": query})
                st.session_state.chat_history.append({"role": "agent", "content": response})
                if hasattr(st.session_state, 'current_query'):
                    delattr(st.session_state, 'current_query')
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

def show_rag_agent_page(api_key):
    st.markdown('<h2 class="sub-header">📚 RAG-Enhanced Agent</h2>', unsafe_allow_html=True)
    
    if st.session_state.rag_agent is None:
        try:
            with st.spinner("Initializing RAG Agent..."):
                st.session_state.rag_agent = RAGAgent(api_key)
            st.success("✅ RAG Agent initialized!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.error(traceback.format_exc())
            return
    
    # Document upload
    st.markdown("### 📁 Document Upload")
    uploaded_file = st.file_uploader("Upload text file", type=['txt'])
    
    if uploaded_file:
        try:
            content = uploaded_file.read().decode('utf-8')
            with st.spinner("Processing document..."):
                success, total = st.session_state.rag_agent.add_document(content, uploaded_file.name)
            st.success(f"✅ Processed: {success}/{total} chunks")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    if st.button("Add Sample Documents"):
        sample_content = """
        Company Policy: Remote work is allowed for all employees.
        Working hours: 9 AM to 5 PM local time.
        Weekly reports due every Friday.
        API Authentication: Use JWT tokens for authentication.
        Token expires after 24 hours.
        Rate limit: 1000 requests per hour per user.
        """
        
        try:
            with st.spinner("Adding sample documents..."):
                success, total = st.session_state.rag_agent.add_document(sample_content, "Sample Policy")
            st.success(f"✅ Added sample: {success}/{total} chunks")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Chat
    for message in st.session_state.rag_chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**RAG Agent:** {message['content']}")
    
    rag_query = st.text_input("Ask about documents:")
    
    if st.button("Send Query") and rag_query:
        with st.spinner("Searching and generating response..."):
            try:
                response = st.session_state.rag_agent.chat(rag_query)
                st.session_state.rag_chat_history.append({"role": "user", "content": rag_query})
                st.session_state.rag_chat_history.append({"role": "agent", "content": response})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if st.button("Clear RAG Chat"):
        st.session_state.rag_chat_history = []
        st.rerun()

if __name__ == "__main__":
    main()
