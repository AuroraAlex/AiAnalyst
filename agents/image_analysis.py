"""
Image Analysis Agent implementation using LangChain's OpenAI integration.

This module provides the core functionality for analyzing images using
large language models and custom tools.
"""

from typing import List, Dict, Any
import base64
from pathlib import Path
import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from tools.image_tool import ImageTool

class ImageAnalysisAgent:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.chat_history = []
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the LangChain chat model with configuration."""
        api_keys = self.config.get("api_keys", {})
        model_config = self.config.get("models", {}).get("image_analysis", {})
        
        self.llm = ChatOpenAI(
            model_name=model_config.get("model_name", "qwen-vl-max-latest"),
            openai_api_key=api_keys.get("api_key"),  # 更新为新的键名
            openai_api_base=model_config.get("base_url")
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
    def initialize(self):
        """Initialize the agent with necessary tools."""
        self.image_tool = ImageTool()
        
    def analyze_image(self, image_path: str, query: str = None) -> str:
        """Analyze the given image and return insights."""
        try:
            # Load and process the image
            image = self.image_tool.load_image(image_path)
            image = self.image_tool.resize_image(image)
            
            # Convert image to base64
            image_data = self.image_tool.convert_to_base64(image_path)
            
            # Create messages for the API
            messages = [
                SystemMessage(content=[{
                    "type": "text", 
                    "text": "You are an AI assistant that can see and analyze images. Provide detailed and accurate descriptions of images."
                }]),
                HumanMessage(content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": query if query else "请详细描述这张图片。"
                    }
                ])
            ]
            
            # Get response from model
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
        
    def chat(self, message: str, image_path: str = None) -> str:
        """Handle chat interaction with memory of conversation history."""
        try:
            if image_path:
                # If image is provided, analyze it first
                analysis = self.analyze_image(image_path, message)
                self.chat_history.append(HumanMessage(
                    content=[{"type": "text", "text": f"{message} (with image: {image_path})"}]
                ))
                self.chat_history.append(AIMessage(
                    content=[{"type": "text", "text": analysis}]
                ))
                return analysis
                
            # Handle regular chat message
            messages = [
                SystemMessage(content=[{
                    "type": "text",
                    "text": "You are a helpful assistant."
                }])
            ]
            
            # Add chat history
            messages.extend(self.chat_history)
            
            # Add current message
            messages.append(HumanMessage(
                content=[{"type": "text", "text": message}]
            ))
            
            # Get response from LangChain
            response = self.llm.invoke(messages)
            
            # Update chat history
            self.chat_history.append(HumanMessage(
                content=[{"type": "text", "text": message}]
            ))
            self.chat_history.append(AIMessage(
                content=[{"type": "text", "text": response.content}]
            ))
            
            return response.content

        except Exception as e:
            error_msg = f"Error in chat: {str(e)}"
            self.chat_history.append(SystemMessage(
                content=[{"type": "text", "text": error_msg}]
            ))
            return error_msg

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Return the conversation history."""
        return [
            {
                "role": "user" if isinstance(msg, HumanMessage) else "assistant" if isinstance(msg, AIMessage) else "system",
                "content": msg.content
            }
            for msg in self.chat_history
        ]