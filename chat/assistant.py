import json
import ollama
from typing import Dict, List, Any, Optional
import logging
import re

class ChatRAGAssistant:
    def __init__(self, chat_rag, model_name="deepseek-r1:8b"):
        self.chat_rag = chat_rag
        self.model_name = model_name
        
        # Available functions for the RAG system
        self.available_functions = {
            "combined_search_optimized": self.chat_rag.combined_search_optimized,
            "semantic_search_optimized": self.chat_rag.semantic_search_optimized,
            "keyword_search_optimized": self.chat_rag.keyword_search_optimized,
            "get_stats": self.chat_rag.get_stats,
            "get_user_messages": self.chat_rag.get_user_messages,
            "get_recent_messages": self.chat_rag.get_recent_messages,
            "tfidf_search_optimized": self.chat_rag.tfidf_search_optimized
        }
        
        # Setup minimal logging
        logging.basicConfig(level=logging.WARNING)
        
    def clean_deepseek_response(self, response: str) -> str:
        """Clean DeepSeek response by removing <think> tags and extracting clean content"""
        # Remove <think>...</think> blocks (including multiline)
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Remove any remaining <think> or </think> tags
        cleaned = re.sub(r'</?think>', '', cleaned)
        
        # Clean up extra whitespace
        cleaned = cleaned.strip()
        
        # If the response is mostly empty after cleaning, try to extract JSON from original
        if not cleaned or len(cleaned) < 10:
            # Try to find JSON in the original response
            json_match = re.search(r'\{.*?"function_call".*?\}', response, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)
        
        return cleaned
    
    def extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from response, handling DeepSeek's think tags"""
        cleaned_response = self.clean_deepseek_response(response)
        
        # Try to parse the cleaned response as JSON
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            pass
        
        # If that fails, try to find JSON pattern in the response
        json_patterns = [
            r'\{[^{}]*"function_call"[^{}]*\{[^{}]*\}[^{}]*\}',  # Simple JSON
            r'\{.*?"function_call".*?\}',  # More flexible JSON
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, cleaned_response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        return None

    def get_function_definitions(self) -> List[Dict]:
        """Define available functions for the LLM"""
        return [
            {
                "name": "Combined_search_optimized",
                "description": "Advanced hybrid search combining semantic, keyword, and TF-IDF methods with parallel processing. Best for finding specific topics, events, fights, discussions, etc. but computationallly expensive. **USE WHEN UNCERTAIN**",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "description": "Number of results", "default": 15}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "semantic_search_optimized",
                "description": "Ultra-fast semantic search using precomputed embeddings. Find messages with similar meaning and context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Concept or phrase to search for semantically"},
                        "top_k": {"type": "integer", "description": "Number of results", "default": 15}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "keyword_search_optimized",
                "description": "Optimized keyword search with exact and partial matching. Fast for finding specific words or phrases.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "oneOf": [
                                {"type": "string", "description": "Space-separated keywords"},
                                {"type": "array", "items": {"type": "string"}, "description": "List of keywords"}
                            ],
                            "description": "Keywords to search for"
                        },
                        "top_k": {"type": "integer", "description": "Number of results", "default": 15}
                    },
                    "required": ["keywords"]
                }
            },
            {
                "name": "tfidf_search_optimized",
                "description": "Fast TF-IDF search for finding documents with relevant term frequencies. Good for longer queries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Query text for TF-IDF search"},
                        "top_k": {"type": "integer", "description": "Number of results", "default": 15}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_user_messages",
                "description": "Get messages from a specific user with optimized retrieval",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "username": {"type": "string", "description": "Username to search for"},
                        "limit": {"type": "integer", "description": "Number of messages to retrieve", "default": 10}
                    },
                    "required": ["username"]
                }
            },
            {
                "name": "get_recent_messages",
                "description": "Get recent messages from the last N hours with optimized filtering",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "hours": {"type": "integer", "description": "Hours to look back from current time"},
                        "limit": {"type": "integer", "description": "Number of messages to retrieve", "default": 10}
                    },
                    "required": ["hours"]
                }
            },
            {
                "name": "get_stats",
                "description": "Get comprehensive dataset statistics including message counts, users, date ranges, and common words",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
        ]
    def create_system_prompt(self) -> str:
        """Create system prompt for function calling with DeepSeek-specific instructions"""
        functions_json = json.dumps(self.get_function_definitions(), indent=2)
        
        return f"""You are a Tanglish(Tamil and english) based WhatsApp chat analyzer. When users ask about chat content, you MUST search the chat data first using the available functions, then provide a comprehensive answer based on the results.

Available Functions:
{functions_json}

CRITICAL INSTRUCTIONS:
1. For ANY question about chat content, you MUST call appropriate functions first
2. Use best_search for general queries about topics, events, fights, discussions
3. Use get_user_messages when asked about specific users
4. Use get_recent_messages for recent activity questions
5. Use keyword_search for exact word/phrase searches
6. Use batch_text_similarity_search for finding messages similar to a given text or concept
7. Use get_emotion_similarity for finding emotionally similar messages
8. Use semantic_search_optimized for conceptual similarity searches

IMPORTANT: Do NOT use <think> tags in your response. Respond with ONLY the JSON function call format below:

{{
    "function_call": {{
        "name": "function_name",
        "arguments": {{
            "parameter": "value"
        }}
    }}
}}

After getting function results, provide a natural, helpful answer based on the data. Do not mention the function calls in your final response - just answer the user's question naturally using the information you found."""

    def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute function and return results"""
        if function_name not in self.available_functions:
            return {"error": f"Function '{function_name}' not available"}
        
        try:
            func = self.available_functions[function_name]
            result = func(**arguments)
            return result
        except Exception as e:
            return {"error": f"Function error: {str(e)}"}

    def query(self, user_input: str) -> str:
        """Main query function - handles everything internally and returns clean answer"""
        
        # Step 1: Get initial LLM response (should be function call for chat queries)
        messages = [
            {"role": "system", "content": self.create_system_prompt()},
            {"role": "user", "content": user_input}
        ]
        
        try:
            response = ollama.chat(model=self.model_name, messages=messages)
            assistant_response = response['message']['content'].replace("`","'").strip()
        except Exception as e:
            return f"Error: Unable to process query - {str(e)}"
        
        # Step 2: Check if it's a function call (with DeepSeek handling)
        function_call_data = self.extract_json_from_response(assistant_response)

        
        if function_call_data and 'function_call' in function_call_data:
            # Execute the function
            function_result = self._handle_function_call_data(function_call_data)

            # Step 3: Get final answer based on function results
            messages.append({"role": "assistant", "content": json.dumps(function_call_data)})
            messages.append({
                "role": "user", 
                "content": f"Based on these search results, please answer the original question: '{user_input}'\n\nSearch Results: {json.dumps(function_result, default=str, ensure_ascii=False)}\n\nIMPORTANT: Do NOT use <think> tags. Provide a direct, natural answer."
            })
            
            try:
                final_response = ollama.chat(model=self.model_name, messages=messages)
                final_answer = final_response['message']['content'].replace("`","'")
                # Clean any remaining think tags from final response
                return self.clean_deepseek_response(final_answer)
            except Exception as e:
                return f"Error generating final response: {str(e)}"
        else:
            # Direct response (for non-chat queries) - clean it
            return self.clean_deepseek_response(assistant_response)

    def _is_function_call(self, message: str) -> bool:
        """Check if message is a function call (DeepSeek compatible)"""
        function_call_data = self.extract_json_from_response(message)
        return function_call_data is not None and 'function_call' in function_call_data

    def _handle_function_call(self, message: str) -> Any:
        """Parse and execute function call (DeepSeek compatible)"""
        function_call_data = self.extract_json_from_response(message)
        if function_call_data:
            return self._handle_function_call_data(function_call_data)
        else:
            return {"error": "Could not parse function call from response"}

    def _handle_function_call_data(self, function_call_data: Dict) -> Any:
        """Execute function call from parsed data"""
        try:
            function_call = function_call_data['function_call']
            function_name = function_call['name']
            arguments = function_call.get('arguments', {})
            
            return self.execute_function(function_name, arguments)
        except Exception as e:
            return {"error": f"Function call error: {str(e)}"}

    def chat(self, user_input: str) -> str:
        """Simple chat interface - just returns the answer"""
        return self.query(user_input)