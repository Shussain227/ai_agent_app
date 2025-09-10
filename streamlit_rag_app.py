import streamlit as st
import os
import sys

# CRITICAL: Set this before importing openai
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None) 
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

# Page config
st.set_page_config(
    page_title="AI Agents Demo - Streamlit Cloud Compatible",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import packages
try:
    import openai
    import json
    from datetime import datetime
    from typing import Dict, Any, List
    OPENAI_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing OpenAI: {e}")
    OPENAI_AVAILABLE = False

# Initialize session state
if 'simple_agent' not in st.session_state:
    st.session_state.simple_agent = None
if 'simple_chat' not in st.session_state:
    st.session_state.simple_chat = []

# Alternative OpenAI client creation for Streamlit Cloud
def create_openai_client_v1_3_8(api_key: str):
    """Create OpenAI client compatible with Streamlit Cloud"""
    try:
        # Method 1: Direct instantiation (works with v1.3.8)
        client = openai.OpenAI(api_key=api_key)
        return client
    except Exception as e1:
        try:
            # Method 2: Legacy initialization
            openai.api_key = api_key
            client = openai.OpenAI(api_key=api_key)
            return client
        except Exception as e2:
            # Method 3: Minimal parameters only
            try:
                import openai as openai_module
                client = openai_module.OpenAI(api_key=api_key)
                return client
            except Exception as e3:
                st.error("All OpenAI initialization methods failed:")
                st.error(f"Method 1: {e1}")
                st.error(f"Method 2: {e2}")  
                st.error(f"Method 3: {e3}")
                raise Exception(f"Cannot create OpenAI client: {e3}")

class SimpleTaskAgent:
    def __init__(self, openai_api_key: str):
        self.client = create_openai_client_v1_3_8(openai_api_key)
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather information for a specific location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "The city name or location"}
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
                            "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]
    
    def get_weather(self, location: str) -> Dict[str, Any]:
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
                "location": location, "temperature": data["temp"],
                "description": data["desc"], "humidity": data["humidity"],
                "feels_like": data["temp"] - 2
            }
        else:
            return {
                "location": location, "temperature": 20,
                "description": "clear sky", "humidity": 60, "feels_like": 18
            }
    
    def calculate(self, expression: str) -> Dict[str, Any]:
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
        if function_name == "get_weather":
            return self.get_weather(arguments["location"])
        elif function_name == "calculate":
            return self.calculate(arguments["expression"])
        else:
            return {"error": f"Unknown function: {function_name}"}
    
    def chat(self, user_message: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can check weather and perform calculations."},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
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
                    model="gpt-4-turbo-preview",
                    messages=messages
                )
                return final_response.choices[0].message.content
            else:
                return assistant_message.content
                
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    st.title("AI Agents Demo - Streamlit Cloud Compatible")
    st.markdown("Simple Task Agent with weather and calculator tools")
    
    # System Status  
    st.sidebar.markdown("## System Status")
    if OPENAI_AVAILABLE:
        st.sidebar.success("✅ OpenAI Available")
    else:
        st.sidebar.error("❌ OpenAI Not Available")
    
    # Show OpenAI version info
    with st.sidebar.expander("🔍 System Info"):
        st.text(f"Python: {sys.version_info.major}.{sys.version_info.minor}")
        st.text(f"Streamlit: {st.__version__}")
        try:
            st.text(f"OpenAI: {openai.__version__}")
        except:
            st.text("OpenAI version: Unknown")
    
    # API Key
    st.sidebar.markdown("---")
    st.sidebar.markdown("## API Configuration")
    
    openai_key = None
    try:
        openai_key = st.secrets["OPENAI_API_KEY"]
        st.sidebar.success("✅ OpenAI key from secrets")
    except:
        openai_key = st.sidebar.text_input("OpenAI API Key:", type="password")
        if openai_key:
            st.sidebar.info("API key entered manually")
    
    if not openai_key:
        st.warning("Please enter your OpenAI API key in the sidebar")
        st.info("For Streamlit Cloud: Add OPENAI_API_KEY to your app secrets")
        return
    
    # Initialize agent
    if st.session_state.simple_agent is None:
        try:
            with st.spinner("Initializing Simple Task Agent..."):
                st.session_state.simple_agent = SimpleTaskAgent(openai_key)
            st.success("✅ Simple Task Agent ready!")
        except Exception as e:
            st.error(f"❌ Error initializing agent:")
            st.error(str(e))
            
            # Show detailed error info
            with st.expander("🔍 Error Details"):
                import traceback
                st.code(traceback.format_exc())
                
            # Show troubleshooting tips
            st.markdown("### Troubleshooting Tips:")
            st.markdown("1. Verify your OpenAI API key is correct")
            st.markdown("2. Check if your API key has credits")
            st.markdown("3. Try refreshing the page")
            st.markdown("4. Check the error details above")
            return
    
    # Example buttons
    st.markdown("### 🎯 Try These Examples:")
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
    
    # Chat interface
    for message in st.session_state.simple_chat:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])
    
    # Input handling
    query = st.chat_input("Ask about weather or calculations...")
    
    # Handle button clicks
    if hasattr(st.session_state, 'current_query'):
        query = st.session_state.current_query
        delattr(st.session_state, 'current_query')
    
    if query:
        st.session_state.simple_chat.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    response = st.session_state.simple_agent.chat(query)
                    st.markdown(response)
                    st.session_state.simple_chat.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Chat Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.simple_chat.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.simple_chat = []
        st.rerun()
    
    # Footer info
    st.markdown("---")
    st.markdown("### About This Demo")
    st.info("This is a simplified version optimized for Streamlit Cloud compatibility. It includes the Simple Task Agent with weather and calculator functions.")

if __name__ == "__main__":
    main()
