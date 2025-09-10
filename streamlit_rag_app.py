import streamlit as st
import os
import sys
import traceback
import asyncio
from abc import ABC, abstractmethod

# CRITICAL: Remove proxy environment variables that cause OpenAI client issues
proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
for var in proxy_vars:
    if var in os.environ:
        del os.environ[var]

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
    page_title="AI Agents Demo - Complete Suite",
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
    from typing import Dict, Any, List, Optional, Literal
    from enum import Enum
    OPENAI_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing OpenAI: {e}")
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing Anthropic: {e}")
    ANTHROPIC_AVAILABLE = False

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
        CHROMADB_AVAILABLE = False
    else:
        st.error(f"Runtime error with ChromaDB: {e}")
        CHROMADB_AVAILABLE = False

# Helper function to create OpenAI client safely
def create_openai_client(api_key: str):
    """Create OpenAI client with multiple fallback methods for Streamlit Cloud compatibility"""
    
    # Method 1: Basic initialization (most common)
    try:
        return openai.OpenAI(api_key=api_key)
    except Exception as e1:
        st.warning(f"Standard OpenAI init failed: {e1}")
        
        # Method 2: With explicit timeout and no proxies
        try:
            return openai.OpenAI(
                api_key=api_key,
                timeout=60.0,
                max_retries=2
            )
        except Exception as e2:
            st.warning(f"OpenAI with timeout failed: {e2}")
            
            # Method 3: Legacy style initialization
            try:
                # Set API key globally as fallback
                openai.api_key = api_key
                client = openai.OpenAI(api_key=api_key)
                return client
            except Exception as e3:
                st.error("All OpenAI client initialization methods failed:")
                st.error(f"Method 1 (standard): {e1}")
                st.error(f"Method 2 (timeout): {e2}")
                st.error(f"Method 3 (legacy): {e3}")
                raise Exception(f"Cannot create OpenAI client. Final error: {e3}")

# Helper function to create Anthropic client safely
def create_anthropic_client(api_key: str):
    """Create Anthropic client with error handling"""
    try:
        return anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to create Anthropic client: {e}")
        raise

# Initialize session state for all agents
session_keys = [
    'simple_agent', 'rag_agent', 'research_manager', 'multi_llm_agent',
    'simple_chat', 'rag_chat', 'research_chat', 'multi_llm_chat'
]

for key in session_keys:
    if key not in st.session_state:
        st.session_state[key] = None if 'agent' in key or 'manager' in key else []

# ===================== DEMO 1: SIMPLE TASK AGENT =====================

class SimpleTaskAgent:
    def __init__(self, openai_api_key: str):
        self.client = create_openai_client(openai_api_key)
        
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
                model="gpt-4", messages=messages, tools=self.tools, tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message
            
            if assistant_message.tool_calls:
                messages.append({
                    "role": "assistant", "content": assistant_message.content,
                    "tool_calls": assistant_message.tool_calls
                })
                
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    function_result = self.execute_function(function_name, function_args)
                    
                    messages.append({
                        "role": "tool", "tool_call_id": tool_call.id,
                        "content": json.dumps(function_result)
                    })
                
                final_response = self.client.chat.completions.create(model="gpt-4", messages=messages)
                return final_response.choices[0].message.content
            else:
                return assistant_message.content
                
        except Exception as e:
            return f"Error: {str(e)}"

# ===================== DEMO 2: RAG-ENHANCED AGENT =====================

class RAGAgent:
    def __init__(self, openai_api_key: str):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available")
            
        self.client = create_openai_client(openai_api_key)
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client()
        
        doc_name = f"docs_{uuid.uuid4().hex[:8]}"
        mem_name = f"memory_{uuid.uuid4().hex[:8]}"
        
        self.documents_collection = self.chroma_client.create_collection(
            name=doc_name, metadata={"description": "Document knowledge base"}
        )
        
        self.memory_collection = self.chroma_client.create_collection(
            name=mem_name, metadata={"description": "Conversation memory"}
        )
        
        self.conversation_history = []
        
        self.tools = [{
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
        }]
    
    def get_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(model="text-embedding-ada-002", input=text)
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error getting embedding: {e}")
            return [0.0] * 1536
    
    def split_text(self, text: str, max_tokens: int = 500) -> List[str]:
        tokens = self.encoding.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        return chunks
    
    def add_document(self, content: str, title: str = None):
        doc_id = str(uuid.uuid4())
        chunks = self.split_text(content, max_tokens=500)
        successful_chunks = 0
        
        for i, chunk in enumerate(chunks):
            try:
                embedding = self.get_embedding(chunk)
                self.documents_collection.add(
                    documents=[chunk], embeddings=[embedding],
                    metadatas=[{"title": title or "Document", "chunk_index": i, 
                              "timestamp": datetime.now().isoformat()}],
                    ids=[f"{doc_id}_chunk_{i}"]
                )
                successful_chunks += 1
            except Exception as e:
                st.error(f"Error adding chunk {i}: {e}")
        
        return successful_chunks, len(chunks)
    
    def search_documents(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.get_embedding(query)
            results = self.documents_collection.query(
                query_embeddings=[query_embedding], n_results=max_results
            )
            
            formatted_results = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    formatted_results.append({"content": doc, "metadata": metadata})
            
            return formatted_results
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]
    
    def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if function_name == "search_documents":
            query = arguments["query"]
            max_results = arguments.get("max_results", 3)
            return {"results": self.search_documents(query, max_results)}
        else:
            return {"error": f"Unknown function: {function_name}"}
    
    def chat(self, user_message: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_message})
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant with access to documents. Use search_documents to find relevant information when needed."},
            *self.conversation_history[-10:]
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4", messages=messages, tools=self.tools, tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message
            
            if assistant_message.tool_calls:
                messages.append({
                    "role": "assistant", "content": assistant_message.content,
                    "tool_calls": assistant_message.tool_calls
                })
                
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    function_result = self.execute_function(function_name, function_args)
                    
                    messages.append({
                        "role": "tool", "tool_call_id": tool_call.id,
                        "content": json.dumps(function_result)
                    })
                
                final_response = self.client.chat.completions.create(model="gpt-4", messages=messages)
                final_content = final_response.choices[0].message.content
            else:
                final_content = assistant_message.content
            
            self.conversation_history.append({"role": "assistant", "content": final_content})
            return final_content
            
        except Exception as e:
            return f"Error: {str(e)}"

# ===================== DEMO 3: MULTI-AGENT RESEARCH SYSTEM =====================

class BaseAgent(ABC):
    def __init__(self, name: str, role: str, openai_api_key: str):
        self.name = name
        self.role = role
        self.client = create_openai_client(openai_api_key)
        self.conversation_history = []
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        pass
    
    def process(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": self.format_task(task, context)}
            ]
            
            response = self.client.chat.completions.create(model="gpt-4", messages=messages, temperature=0.7)
            result = response.choices[0].message.content
            
            return {
                "agent": self.name, "task": task, "result": result,
                "timestamp": datetime.now().isoformat(), "status": "success"
            }
        except Exception as e:
            return {
                "agent": self.name, "task": task, "result": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat(), "status": "error"
            }
    
    def format_task(self, task: str, context: Dict[str, Any] = None) -> str:
        formatted_task = f"Task: {task}\n"
        if context:
            formatted_task += f"\nContext from previous agents:\n"
            for key, value in context.items():
                formatted_task += f"- {key}: {value}\n"
        return formatted_task

class DataCollectorAgent(BaseAgent):
    def __init__(self, openai_api_key: str):
        super().__init__("DataCollector", "Data Collection Specialist", openai_api_key)
    
    def get_system_prompt(self) -> str:
        return """You are a Data Collection Specialist. Your role is to:
1. Identify what data is needed for research tasks
2. Suggest reliable sources for information
3. Organize and structure raw data
4. Create data collection plans
5. Validate data quality and completeness

When given a research topic, provide:
- List of required data points
- Suggested data sources
- Data collection methodology
- Quality checks to perform
- Timeline for data gathering

Be thorough, systematic, and focus on data reliability."""

class AnalysisAgent(BaseAgent):
    def __init__(self, openai_api_key: str):
        super().__init__("Analyst", "Data Analysis Expert", openai_api_key)
    
    def get_system_prompt(self) -> str:
        return """You are a Data Analysis Expert. Your role is to:
1. Analyze collected data for patterns and insights
2. Perform statistical analysis when appropriate
3. Identify trends, correlations, and anomalies
4. Create analytical frameworks
5. Generate evidence-based conclusions

When analyzing data or research topics:
- Apply appropriate analytical methods
- Look for patterns and relationships
- Consider multiple perspectives
- Validate findings with evidence
- Highlight limitations and assumptions
- Suggest further analysis if needed

Be rigorous, objective, and evidence-based in your analysis."""

class ReportWriterAgent(BaseAgent):
    def __init__(self, openai_api_key: str):
        super().__init__("ReportWriter", "Technical Writing Specialist", openai_api_key)
    
    def get_system_prompt(self) -> str:
        return """You are a Technical Writing Specialist. Your role is to:
1. Create clear, well-structured reports
2. Synthesize information from multiple sources
3. Write for specific audiences
4. Ensure logical flow and coherence
5. Format content professionally

When creating reports:
- Start with executive summary
- Use clear headings and structure
- Present information logically
- Include supporting evidence
- Write in appropriate style for audience
- End with conclusions and recommendations

Focus on clarity, accuracy, and professional presentation."""

class ResearchManager:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.data_collector = DataCollectorAgent(openai_api_key)
        self.analyst = AnalysisAgent(openai_api_key)
        self.report_writer = ReportWriterAgent(openai_api_key)
        self.current_research = None
        self.research_results = {}
    
    def conduct_research(self, research_topic: str, research_type: str = "comprehensive") -> Dict[str, Any]:
        self.current_research = {
            "topic": research_topic, "type": research_type,
            "started_at": datetime.now().isoformat(), "status": "in_progress"
        }
        
        try:
            # Phase 1: Data Collection Planning
            data_task = f"Create a comprehensive data collection plan for researching: {research_topic}"
            data_result = self.data_collector.process(data_task)
            self.research_results["data_collection"] = data_result
            
            # Phase 2: Analysis Framework Development
            analysis_task = f"Design an analysis framework for: {research_topic}"
            analysis_context = {"data_plan": data_result["result"]}
            analysis_result = self.analyst.process(analysis_task, analysis_context)
            self.research_results["analysis"] = analysis_result
            
            # Phase 3: Report Generation
            report_task = f"Create a comprehensive research report on: {research_topic}"
            report_context = {
                "data_collection_plan": data_result["result"],
                "analysis_framework": analysis_result["result"]
            }
            report_result = self.report_writer.process(report_task, report_context)
            self.research_results["report"] = report_result
            
            # Phase 4: Final Synthesis
            synthesis = self.synthesize_results()
            
            self.current_research["status"] = "completed"
            self.current_research["completed_at"] = datetime.now().isoformat()
            
            return {
                "research_info": self.current_research,
                "phases": self.research_results,
                "synthesis": synthesis,
                "success": True
            }
            
        except Exception as e:
            self.current_research["status"] = "failed"
            self.current_research["error"] = str(e)
            return {
                "research_info": self.current_research,
                "phases": self.research_results,
                "error": str(e), "success": False
            }
    
    def synthesize_results(self) -> Dict[str, Any]:
        synthesis_prompt = f"""
        Synthesize the following research work into key insights and recommendations:
        
        RESEARCH TOPIC: {self.current_research['topic']}
        
        DATA COLLECTION PLAN:
        {self.research_results['data_collection']['result']}
        
        ANALYSIS FRAMEWORK:
        {self.research_results['analysis']['result']}
        
        RESEARCH REPORT:
        {self.research_results['report']['result']}
        
        Please provide:
        1. Key insights from the multi-agent research process
        2. Main recommendations
        3. Next steps for implementation
        4. Quality assessment of the research approach
        """
        
        try:
            client = create_openai_client(self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a research director synthesizing work from multiple specialist agents."},
                    {"role": "user", "content": synthesis_prompt}
                ]
            )
            
            return {
                "synthesis": response.choices[0].message.content,
                "timestamp": datetime.now().isoformat(), "status": "success"
            }
        except Exception as e:
            return {
                "synthesis": f"Synthesis failed: {str(e)}",
                "timestamp": datetime.now().isoformat(), "status": "error"
            }

# ===================== DEMO 4: MULTI-LLM AGENT SYSTEM =====================

class ModelProvider(Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    BOTH = "both"

class TaskType(Enum):
    CREATIVE_WRITING = "creative_writing"
    ANALYSIS = "analysis"
    CODING = "coding"
    REASONING = "reasoning"
    GENERAL = "general"

class MultiLLMAgent:
    def __init__(self, openai_api_key: str, claude_api_key: str = None):
        self.openai_client = create_openai_client(openai_api_key)
        
        if claude_api_key and ANTHROPIC_AVAILABLE:
            try:
                self.claude_client = create_anthropic_client(claude_api_key)
                self.claude_available = True
            except Exception as e:
                st.warning(f"Claude client creation failed: {e}")
                self.claude_client = None
                self.claude_available = False
        else:
            self.claude_client = None
            self.claude_available = False
        
        self.model_preferences = {
            TaskType.CREATIVE_WRITING: ModelProvider.CLAUDE if self.claude_available else ModelProvider.OPENAI,
            TaskType.ANALYSIS: ModelProvider.CLAUDE if self.claude_available else ModelProvider.OPENAI,
            TaskType.CODING: ModelProvider.OPENAI,
            TaskType.REASONING: ModelProvider.BOTH if self.claude_available else ModelProvider.OPENAI,
            TaskType.GENERAL: ModelProvider.BOTH if self.claude_available else ModelProvider.OPENAI
        }
        
        self.conversation_history = []
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "compare_models",
                    "description": "Get responses from both Claude and OpenAI for comparison",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "The prompt to send to both models"},
                            "task_type": {
                                "type": "string",
                                "enum": ["creative_writing", "analysis", "coding", "reasoning", "general"],
                                "description": "Type of task for optimal model selection"
                            }
                        },
                        "required": ["prompt"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "select_best_model",
                    "description": "Analyze task and recommend the best model",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_description": {"type": "string", "description": "Description of the task to analyze"}
                        },
                        "required": ["task_description"]
                    }
                }
            }
        ]
    
    def classify_task(self, task_description: str) -> TaskType:
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ["write", "story", "poem", "creative", "narrative"]):
            return TaskType.CREATIVE_WRITING
        elif any(word in task_lower for word in ["analyze", "analysis", "compare", "evaluate", "assess"]):
            return TaskType.ANALYSIS
        elif any(word in task_lower for word in ["code", "program", "function", "algorithm", "debug"]):
            return TaskType.CODING
        elif any(word in task_lower for word in ["logic", "reasoning", "solve", "problem", "think"]):
            return TaskType.REASONING
        else:
            return TaskType.GENERAL
    
    def call_openai(self, prompt: str, system_message: str = None) -> Dict[str, Any]:
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4", messages=messages, temperature=0.7
            )
            
            return {
                "provider": "OpenAI", "model": "gpt-4",
                "response": response.choices[0].message.content,
                "timestamp": datetime.now().isoformat(), "status": "success"
            }
        except Exception as e:
            return {
                "provider": "OpenAI", "model": "gpt-4",
                "response": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat(), "status": "error"
            }
    
    def call_claude(self, prompt: str, system_message: str = None) -> Dict[str, Any]:
        if not self.claude_available:
            return {
                "provider": "Claude", "model": "claude-3-sonnet-20240229",
                "response": "Claude API not available",
                "timestamp": datetime.now().isoformat(), "status": "error"
            }
        
        try:
            messages = [{"role": "user", "content": prompt}]
            kwargs = {"model": "claude-3-sonnet-20240229", "max_tokens": 1000, "messages": messages}
            
            if system_message:
                kwargs["system"] = system_message
            
            response = self.claude_client.messages.create(**kwargs)
            
            return {
                "provider": "Claude", "model": "claude-3-sonnet-20240229",
                "response": response.content[0].text,
                "timestamp": datetime.now().isoformat(), "status": "success"
            }
        except Exception as e:
            return {
                "provider": "Claude", "model": "claude-3-sonnet-20240229",
                "response": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat(), "status": "error"
            }
    
    def compare_models(self, prompt: str, task_type: str = "general") -> Dict[str, Any]:
        task_enum = TaskType(task_type) if task_type in [t.value for t in TaskType] else TaskType.GENERAL
        
        system_messages = {
            TaskType.CREATIVE_WRITING: "You are a creative writing assistant. Focus on engaging, imaginative content.",
            TaskType.ANALYSIS: "You are an analytical expert. Provide thorough, structured analysis.",
            TaskType.CODING: "You are a programming expert. Provide clean, efficient, well-documented code.",
            TaskType.REASONING: "You are a logical reasoning expert. Think step-by-step and show your work.",
            TaskType.GENERAL: "You are a helpful assistant. Provide clear, accurate, and useful responses."
        }
        
        system_message = system_messages.get(task_enum, system_messages[TaskType.GENERAL])
        
        openai_result = self.call_openai(prompt, system_message=system_message)
        claude_result = self.call_claude(prompt, system_message=system_message)
        
        return {
            "prompt": prompt, "task_type": task_type,
            "openai_response": openai_result,
            "claude_response": claude_result,
            "comparison_timestamp": datetime.now().isoformat()
        }
    
    def select_best_model(self, task_description: str) -> Dict[str, Any]:
        task_type = self.classify_task(task_description)
        preferred_provider = self.model_preferences[task_type]
        
        recommendations = {
            ModelProvider.OPENAI: {
                "provider": "OpenAI", "model": "gpt-4",
                "strengths": ["Coding", "Structured reasoning", "API integrations", "Function calling"],
                "best_for": "Technical tasks, programming, structured problem solving"
            },
            ModelProvider.CLAUDE: {
                "provider": "Claude", "model": "claude-3-sonnet-20240229",
                "strengths": ["Analysis", "Creative writing", "Long-form content", "Nuanced reasoning"],
                "best_for": "Analysis, creative tasks, detailed explanations"
            },
            ModelProvider.BOTH: {
                "provider": "Both (Comparison)", "model": "Multiple models",
                "strengths": ["Consensus building", "Cross-validation", "Diverse perspectives"],
                "best_for": "Critical decisions, complex analysis, validation"
            }
        }
        
        recommendation = recommendations[preferred_provider]
        
        return {
            "task_description": task_description,
            "classified_type": task_type.value,
            "recommended_provider": preferred_provider.value,
            "recommendation": recommendation,
            "reasoning": f"Based on task type '{task_type.value}', {recommendation['provider']} is recommended for: {recommendation['best_for']}"
        }
    
    def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if function_name == "compare_models":
            return self.compare_models(arguments["prompt"], arguments.get("task_type", "general"))
        elif function_name == "select_best_model":
            return self.select_best_model(arguments["task_description"])
        else:
            return {"error": f"Unknown function: {function_name}"}
    
    def chat(self, user_message: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_message})
        
        system_prompt = """You are an AI agent coordinator that can use both OpenAI GPT-4 and Claude models.

Your capabilities:
1. compare_models: Get responses from both Claude and OpenAI for comparison
2. select_best_model: Recommend the best model for a specific task

When users ask questions:
- For simple queries, answer directly using your knowledge
- For complex or important queries, consider using compare_models to get multiple perspectives
- For tasks requiring model selection, use select_best_model first
- Be intelligent about tool usage - not every query needs multi-model comparison."""

        messages = [
            {"role": "system", "content": system_prompt},
            *self.conversation_history[-6:]
        ]
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4", messages=messages, tools=self.tools, tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message
            
            if assistant_message.tool_calls:
                messages.append({
                    "role": "assistant", "content": assistant_message.content,
                    "tool_calls": assistant_message.tool_calls
                })
                
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    function_result = self.execute_function(function_name, function_args)
                    
                    messages.append({
                        "role": "tool", "tool_call_id": tool_call.id,
                        "content": json.dumps(function_result, indent=2)
                    })
                
                final_response = self.openai_client.chat.completions.create(model="gpt-4", messages=messages)
                final_content = final_response.choices[0].message.content
            else:
                final_content = assistant_message.content
            
            self.conversation_history.append({"role": "assistant", "content": final_content})
            return final_content
            
        except Exception as e:
            error_message = f"Error in multi-LLM agent: {str(e)}"
            self.conversation_history.append({"role": "assistant", "content": error_message})
            return error_message

# ===================== MAIN STREAMLIT APP =====================

def main():
    st.title("AI Agents Demo - Complete Suite")
    st.markdown("Four advanced AI agent systems demonstrating different coordination patterns")
    
    # Sidebar - System Status
    st.sidebar.markdown("## System Status")
    
    status_items = [
        ("OpenAI", OPENAI_AVAILABLE),
        ("Anthropic (Claude)", ANTHROPIC_AVAILABLE),
        ("ChromaDB", CHROMADB_AVAILABLE)
    ]
    
    for name, available in status_items:
        if available:
            st.sidebar.success(f"✅ {name}")
        else:
            st.sidebar.error(f"❌ {name}")
    
    # Debug info
    with st.sidebar.expander("🔍 Debug Information"):
        st.text(f"Python: {sys.version}")
        st.text(f"Streamlit: {st.__version__}")
        
        # Show if proxy vars were found
        proxy_found = any(var in os.environ for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'])
        if proxy_found:
            st.warning("Proxy variables detected (may cause OpenAI issues)")
        else:
            st.success("No proxy variables detected")
    
    # API Keys
    st.sidebar.markdown("---")
    st.sidebar.markdown("## API Configuration")
    
    # Try to get from secrets first
    openai_key = None
    claude_key = None
    
    try:
        openai_key = st.secrets["OPENAI_API_KEY"]
        st.sidebar.success("✅ OpenAI key from secrets")
    except:
        openai_key = st.sidebar.text_input("OpenAI API Key:", type="password")
        if openai_key:
            st.sidebar.info("OpenAI key entered manually")
    
    try:
        claude_key = st.secrets["CLAUDE_API_KEY"]
        st.sidebar.success("✅ Claude key from secrets")
    except:
        claude_key = st.sidebar.text_input("Claude API Key (optional):", type="password")
        if claude_key:
            st.sidebar.info("Claude key entered manually")
    
    if not openai_key:
        st.warning("Please enter your OpenAI API key to use the demos.")
        st.info("💡 For Streamlit Cloud: Add `OPENAI_API_KEY = \"your_key\"` to your app secrets")
        return
    
    # Navigation
    st.sidebar.markdown("---")
    
    demo_options = [
        "Home",
        "1️⃣ Simple Task Agent",
        "2️⃣ RAG-Enhanced Agent" + ("" if CHROMADB_AVAILABLE else " (Disabled)"),
        "3️⃣ Multi-Agent Research",
        "4️⃣ Multi-LLM System" + ("" if ANTHROPIC_AVAILABLE else " (Limited)")
    ]
    
    selected_demo = st.sidebar.radio("Choose Demo:", demo_options)
    
    # Route to appropriate demo
    if selected_demo == "Home":
        show_home_page()
    elif "Simple Task Agent" in selected_demo:
        show_simple_agent_demo(openai_key)
    elif "RAG-Enhanced Agent" in selected_demo:
        if CHROMADB_AVAILABLE:
            show_rag_agent_demo(openai_key)
        else:
            st.error("ChromaDB not available. Please check Python version and SQLite compatibility.")
            st.info("Try: Change Python version to 3.11 in Advanced Settings")
    elif "Multi-Agent Research" in selected_demo:
        show_research_system_demo(openai_key)
    elif "Multi-LLM System" in selected_demo:
        show_multi_llm_demo(openai_key, claude_key)

def show_home_page():
    st.markdown("## Welcome to the Complete AI Agents Demo Suite!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 1️⃣ Simple Task Agent
        **Basic AI agent with tool use**
        - Weather information (mock data)
        - Mathematical calculations
        - OpenAI function calling demonstration
        - Perfect for learning agent basics
        """)
        
        st.markdown("""
        ### 3️⃣ Multi-Agent Research System
        **Coordinated specialist agents**
        - DataCollector Agent (research planning)
        - Analysis Agent (data analysis)
        - ReportWriter Agent (documentation)
        - ResearchManager (orchestration)
        """)
    
    with col2:
        st.markdown("""
        ### 2️⃣ RAG-Enhanced Agent
        **Document knowledge + memory**
        - Upload and process documents
        - Semantic search capabilities
        - Conversation memory
        - Vector database integration
        """)
        
        st.markdown("""
        ### 4️⃣ Multi-LLM System
        **OpenAI + Claude coordination**
        - Intelligent model selection
        - Side-by-side comparison
        - Consensus building
        - Task-specific routing
        """)
    
    # System requirements
    st.markdown("---")
    st.markdown("### System Requirements")
    
    requirements = [
        ("All demos", "OpenAI API key"),
        ("RAG Agent", "ChromaDB (SQLite 3.35.0+)"),
        ("Multi-LLM", "Claude API key (optional)")
    ]
    
    for demo, req in requirements:
        st.markdown(f"**{demo}:** {req}")

def show_simple_agent_demo(openai_key):
    st.markdown("## 1️⃣ Simple Task Agent")
    st.markdown("Basic AI agent with weather and calculator tools")
    
    if st.session_state.simple_agent is None:
        try:
            with st.spinner("Initializing Simple Task Agent..."):
                st.session_state.simple_agent = SimpleTaskAgent(openai_key)
            st.success("✅ Simple Task Agent ready!")
        except Exception as e:
            st.error(f"❌ Error initializing agent:")
            st.error(str(e))
            
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
            return
    
    # Example buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🌤️ Weather in Tokyo"):
            st.session_state.current_simple_query = "What's the weather like in Tokyo?"
    with col2:
        if st.button("🔢 Calculate 15 * 8 + 32"):
            st.session_state.current_simple_query = "Calculate 15 * 8 + 32"
    with col3:
        if st.button("🌦️➕ Weather + Math"):
            st.session_state.current_simple_query = "Weather in London and calculate sqrt(144)"
    
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
    
    if hasattr(st.session_state, 'current_simple_query'):
        query = st.session_state.current_simple_query
        delattr(st.session_state, 'current_simple_query')
    
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
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.simple_chat.append({"role": "assistant", "content": error_msg})
    
    if st.sidebar.button("Clear Simple Chat"):
        st.session_state.simple_chat = []
        st.rerun()

def show_rag_agent_demo(openai_key):
    st.markdown("## 2️⃣ RAG-Enhanced Agent")
    st.markdown("AI agent with document knowledge and conversation memory")
    
    if st.session_state.rag_agent is None:
        try:
            with st.spinner("Initializing RAG Agent..."):
                st.session_state.rag_agent = RAGAgent(openai_key)
            st.success("✅ RAG Agent ready!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return
    
    # Document management
    st.markdown("### 📄 Document Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload text document", type=['txt'])
        if uploaded_file:
            try:
                content = uploaded_file.read().decode('utf-8')
                with st.spinner("Processing document..."):
                    success, total = st.session_state.rag_agent.add_document(content, uploaded_file.name)
                st.success(f"✅ Added {success}/{total} chunks from {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing document: {e}")
    
    with col2:
        if st.button("📝 Add Sample Docs"):
            sample_content = """
            Company Remote Work Policy:
            All employees are eligible for remote work arrangements.
            Working hours: 9 AM - 5 PM local time.
            Weekly reports due every Friday.
            
            API Authentication Guide:
            Use JWT tokens for authentication.
            Tokens expire after 24 hours.
            Rate limit: 1000 requests per hour per user.
            """
            
            try:
                with st.spinner("Adding sample documents..."):
                    success, total = st.session_state.rag_agent.add_document(sample_content, "Sample Company Docs")
                st.success(f"✅ Added {success}/{total} sample chunks")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("### 💬 Chat with Documents")
    
    # Chat interface
    for message in st.session_state.rag_chat:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])
    
    # Example questions
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("❓ Ask about policies"):
            st.session_state.current_rag_query = "What is the company's remote work policy?"
    with col2:
        if st.button("🔍 Ask about API"):
            st.session_state.current_rag_query = "How do I authenticate with the API?"
    with col3:
        if st.button("📊 Ask about limits"):
            st.session_state.current_rag_query = "What are the API rate limits?"
    
    # Input handling
    query = st.chat_input("Ask questions about your documents...")
    
    if hasattr(st.session_state, 'current_rag_query'):
        query = st.session_state.current_rag_query
        delattr(st.session_state, 'current_rag_query')
    
    if query:
        st.session_state.rag_chat.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                try:
                    response = st.session_state.rag_agent.chat(query)
                    st.markdown(response)
                    st.session_state.rag_chat.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.rag_chat.append({"role": "assistant", "content": error_msg})
    
    if st.sidebar.button("Clear RAG Chat"):
        st.session_state.rag_chat = []
        st.rerun()

def show_research_system_demo(openai_key):
    st.markdown("## 3️⃣ Multi-Agent Research System")
    st.markdown("Coordinated specialist agents working together on research tasks")
    
    if st.session_state.research_manager is None:
        try:
            with st.spinner("Initializing Research System..."):
                st.session_state.research_manager = ResearchManager(openai_key)
            st.success("✅ Multi-Agent Research System ready!")
            st.info("🤖 Agents: DataCollector, Analyst, ReportWriter")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return
    
    # Research topic input
    st.markdown("### 🔬 Research Configuration")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        research_topic = st.text_input(
            "Research Topic:",
            placeholder="e.g., Impact of remote work on employee productivity"
        )
    
    with col2:
        research_type = st.selectbox("Research Type:", ["comprehensive", "focused", "exploratory"])
    
    # Example topics
    st.markdown("**Example topics:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏠 Remote Work Impact"):
            st.session_state.current_research_topic = "The impact of remote work on employee productivity and company culture"
    
    with col2:
        if st.button("♻️ Sustainable Energy"):
            st.session_state.current_research_topic = "Sustainable energy solutions for small businesses"
    
    with col3:
        if st.button("🏥 AI in Healthcare"):
            st.session_state.current_research_topic = "AI adoption challenges in healthcare organizations"
    
    # Handle button clicks
    if hasattr(st.session_state, 'current_research_topic'):
        research_topic = st.session_state.current_research_topic
        delattr(st.session_state, 'current_research_topic')
    
    # Conduct research
    if st.button("🚀 Start Multi-Agent Research", disabled=not research_topic):
        if research_topic:
            with st.spinner("🤖 Multi-agent research in progress..."):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulate progress updates
                    progress_updates = [
                        (25, "📊 DataCollector: Planning data collection..."),
                        (50, "🔍 Analyst: Developing analysis framework..."),
                        (75, "📝 ReportWriter: Creating comprehensive report..."),
                        (100, "🎯 ResearchManager: Synthesizing results...")
                    ]
                    
                    for progress, message in progress_updates:
                        progress_bar.progress(progress)
                        status_text.text(message)
                        # Add small delay for demo purposes
                        import time
                        time.sleep(0.5)
                    
                    results = st.session_state.research_manager.conduct_research(research_topic, research_type)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if results["success"]:
                        st.success("✅ Multi-agent research completed!")
                        
                        # Display results
                        st.markdown("### 📋 Research Results")
                        
                        # Executive Summary
                        with st.expander("🎯 Executive Summary", expanded=True):
                            st.markdown(results["synthesis"]["synthesis"])
                        
                        # Individual Agent Contributions
                        with st.expander("🤖 Agent Contributions"):
                            for phase, result in results["phases"].items():
                                st.markdown(f"**{result['agent']}** ({phase.replace('_', ' ').title()}):")
                                st.markdown(f"Status: {result['status']}")
                                with st.expander(f"View {result['agent']} Results"):
                                    st.markdown(result["result"])
                                st.markdown("---")
                        
                        # Research Metadata
                        with st.expander("ℹ️ Research Metadata"):
                            info = results["research_info"]
                            st.markdown(f"**Topic:** {info['topic']}")
                            st.markdown(f"**Type:** {info['type']}")
                            st.markdown(f"**Status:** {info['status']}")
                            st.markdown(f"**Started:** {info['started_at']}")
                            if 'completed_at' in info:
                                st.markdown(f"**Completed:** {info['completed_at']}")
                    
                    else:
                        st.error(f"❌ Research failed: {results.get('error', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"❌ Error during research: {str(e)}")

def show_multi_llm_demo(openai_key, claude_key):
    st.markdown("## 4️⃣ Multi-LLM Agent System")
    st.markdown("Intelligent coordination between OpenAI and Claude models")
    
    # Status indicators
    col1, col2 = st.columns(2)
    with col1:
        if OPENAI_AVAILABLE and openai_key:
            st.success("✅ OpenAI GPT-4 Available")
        else:
            st.error("❌ OpenAI Not Available")
    
    with col2:
        if ANTHROPIC_AVAILABLE and claude_key:
            st.success("✅ Claude Available")
        else:
            st.warning("⚠️ Claude Not Available (Limited functionality)")
    
    if st.session_state.multi_llm_agent is None:
        try:
            with st.spinner("Initializing Multi-LLM System..."):
                st.session_state.multi_llm_agent = MultiLLMAgent(openai_key, claude_key)
            
            if st.session_state.multi_llm_agent.claude_available:
                st.success("✅ Multi-LLM System ready! (Full functionality)")
            else:
                st.warning("⚠️ Multi-LLM System ready (OpenAI only)")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return
    
    # Example queries
    st.markdown("### 🎯 Example Multi-LLM Queries")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🤔 Model Selection"):
            st.session_state.current_multi_query = "Which model should I use for writing a creative short story?"
        
        if st.button("⚖️ Model Comparison"):
            st.session_state.current_multi_query = "Compare how both models would explain quantum computing to a 10-year-old"
    
    with col2:
        if st.button("📊 Analysis Task"):
            st.session_state.current_multi_query = "What are the pros and cons of remote work? I want perspectives from both models."
        
        if st.button("💻 Coding Task"):
            st.session_state.current_multi_query = "Help me debug this Python function that's not working properly"
    
    # Chat interface
    for message in st.session_state.multi_llm_chat:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])
    
    # Input handling
    query = st.chat_input("Ask about model selection, comparisons, or any complex task...")
    
    if hasattr(st.session_state, 'current_multi_query'):
        query = st.session_state.current_multi_query
        delattr(st.session_state, 'current_multi_query')
    
    if query:
        st.session_state.multi_llm_chat.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("🤖 Multi-LLM coordination in progress..."):
                try:
                    response = st.session_state.multi_llm_agent.chat(query)
                    st.markdown(response)
                    st.session_state.multi_llm_chat.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.multi_llm_chat.append({"role": "assistant", "content": error_msg})
    
    if st.sidebar.button("Clear Multi-LLM Chat"):
        st.session_state.multi_llm_chat = []
        st.rerun()

if __name__ == "__main__":
    main()
