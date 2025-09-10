import streamlit as st
import sys
import traceback

# Fix SQLite compatibility issue for ChromaDB
try:
    import sqlite3
    if sqlite3.sqlite_version_info < (3, 35, 0):
        try:
            __import__('pysqlite3')
            sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        except ImportError:
            pass
except ImportError:
    pass

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
    st.error(f"Error importing OpenAI: {e}")
    OPENAI_AVAILABLE = False

try:
    import chromadb
    import tiktoken
    CHROMADB_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing ChromaDB/Tiktoken: {e}")
    CHROMADB_AVAILABLE = False
except RuntimeError as e:
    if "sqlite3" in str(e):
        st.error("SQLite version compatibility issue detected.")
        st.info("This is a known issue with ChromaDB on some platforms.")
        CHROMADB_AVAILABLE = False
    else:
        st.error(f"Runtime error with ChromaDB: {e}")
        CHROMADB_AVAILABLE = False

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

def main():
    st.title("AI Agents Demo")
    
    # Check system status
    st.sidebar.markdown("## System Status")
    
    if OPENAI_AVAILABLE:
        st.sidebar.success("✅ OpenAI Available")
    else:
        st.sidebar.error("❌ OpenAI Not Available")
    
    if CHROMADB_AVAILABLE:
        st.sidebar.success("✅ ChromaDB Available")
        st.sidebar.info("RAG features enabled")
    else:
        st.sidebar.warning("⚠️ ChromaDB Not Available")
        st.sidebar.info("Only Simple Agent available")
    
    # Get API key from Streamlit secrets or user input
    api_key = None
    
    # Try to get from secrets first (for Streamlit Cloud)
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.sidebar.success("✅ API key loaded from secrets")
    except:
        # Fallback to user input
        api_key = st.sidebar.text_input("OpenAI API Key:", type="password")
        if api_key:
            st.sidebar.info("API key entered manually")
    
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar, or configure it in Streamlit Cloud secrets.")
        st.info("For Streamlit Cloud, add `OPENAI_API_KEY = \"your_key_here\"` to your app secrets.")
        return
    
    # Navigation
    if CHROMADB_AVAILABLE:
        page = st.sidebar.radio("Choose Demo:", ["Home", "Simple Agent", "RAG Agent"])
    else:
        page = st.sidebar.radio("Choose Demo:", ["Home", "Simple Agent"])
        st.sidebar.info("RAG Agent disabled due to ChromaDB issues")
    
    # Pages
    if page == "Home":
        show_home_page()
    elif page == "Simple Agent":
        show_simple_agent_page(api_key)
    elif page == "RAG Agent" and CHROMADB_AVAILABLE:
        show_rag_agent_page(api_key)

def show_home_page():
    st.markdown("## Welcome to the AI Agents Demo!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🛠️ Simple Task Agent
        Basic AI agent capabilities:
        - Weather information (mock data)
        - Mathematical calculations
        - OpenAI function calling demo
        """)
    
    with col2:
        if CHROMADB_AVAILABLE:
            st.markdown("""
            ### 📚 RAG-Enhanced Agent
            Advanced knowledge capabilities:
            - Document upload and processing
            - Semantic search
            - Information synthesis
            """)
        else:
            st.markdown("""
            ### 📚 RAG-Enhanced Agent
            ⚠️ **Currently unavailable**
            
            ChromaDB compatibility issue detected. 
            Try changing Python version to 3.11 in Advanced Settings.
            """)

def show_simple_agent_page(api_key):
    st.markdown("## Simple Task Agent")
    st.markdown("This agent can check weather and perform calculations using OpenAI function calling.")
    
    if st.session_state.simple_agent is None:
        try:
            with st.spinner("Initializing Simple Task Agent..."):
                st.session_state.simple_agent = SimpleTaskAgent(api_key)
            st.success("✅ Agent initialized!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return
    
    # Example buttons
    st.markdown("### Try these examples:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌤️ Weather in Tokyo"):
            st.session_state.current_query = "What's the weather like in Tokyo?"
    
    with col2:
        if st.button("🔢 Calculate 15 * 8 + 32"):
            st.session_state.current_query = "Calculate 15 * 8 + 32"
    
    with col3:
        if st.button("🌦️➕ Weather + Math"):
            st.session_state.current_query = "What's the weather in London and calculate sqrt(144)?"
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])
    
    # Handle input
    query = st.chat_input("Ask me about weather or calculations...")
    
    # Handle button clicks
    if hasattr(st.session_state, 'current_query'):
        query = st.session_state.current_query
        delattr(st.session_state, 'current_query')
    
    if query:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.simple_agent.chat(query)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

def show_rag_agent_page(api_key):
    st.markdown("## RAG-Enhanced Agent")
    st.info("RAG functionality would be implemented here when ChromaDB is working.")
    st.markdown("Currently showing placeholder due to ChromaDB compatibility issues.")

if __name__ == "__main__":
    main()
