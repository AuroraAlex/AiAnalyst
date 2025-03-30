from typing import Dict, List, Any, Type, Callable, Optional, Union
import inspect
from pydantic import BaseModel, create_model
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.tools import BaseTool


class LangGraphToolConverter:
    """
    将function calling格式转换为LangGraph中可用的tools
    
    这个类提供了将function calling格式的函数定义转换为LangGraph工作流可使用的工具的功能。
    支持从函数、JSON定义或OpenAI格式转换为LangGraph工具。
    """
    
    def __init__(self):
        self.tools = []
        
    def function_to_tool(self, func: Callable) -> BaseTool:
        """
        将Python函数转换为LangGraph工具
        
        Args:
            func: 要转换的Python函数
            
        Returns:
            转换后的LangGraph工具
        """
        # 使用langchain的tool装饰器进行转换
        tool_func = tool(func)
        self.tools.append(tool_func)
        return tool_func
    
    def functions_to_tools(self, functions: List[Callable]) -> List[BaseTool]:
        """
        批量将Python函数转换为LangGraph工具
        
        Args:
            functions: 要转换的Python函数列表
            
        Returns:
            转换后的LangGraph工具列表
        """
        tools = []
        for func in functions:
            tool_func = self.function_to_tool(func)
            tools.append(tool_func)
        return tools
    
    def json_schema_to_tool(self, name: str, schema: Dict, func: Callable) -> BaseTool:
        """
        从JSON Schema定义转换为LangGraph工具
        
        Args:
            name: 工具名称
            schema: JSON Schema定义
            func: 实际执行的函数
            
        Returns:
            转换后的LangGraph工具
        """
        # 创建pydantic模型
        properties = schema.get("properties", {})
        fields = {}
        
        for prop_name, prop_details in properties.items():
            prop_type = self._get_type_from_schema(prop_details)
            default = prop_details.get("default", ...)
            fields[prop_name] = (prop_type, default)
        
        # 创建参数模型
        params_model = create_model(f"{name}Params", **fields)
        
        # 使用装饰器创建工具
        @tool(name=name, description=schema.get("description", ""))
        def dynamic_tool(**kwargs):
            return func(**kwargs)
        
        dynamic_tool.__annotations__ = {name: params_model for name in fields.keys()}
        
        self.tools.append(dynamic_tool)
        return dynamic_tool
    
    def openai_function_to_tool(self, function_def: Dict, func: Callable) -> BaseTool:
        """
        从OpenAI function calling格式转换为LangGraph工具
        
        Args:
            function_def: OpenAI格式的function定义
            func: 实际执行的函数
            
        Returns:
            转换后的LangGraph工具
        """
        # 提取信息
        name = function_def.get("name", "")
        description = function_def.get("description", "")
        parameters = function_def.get("parameters", {})
        
        # 转换为JSON Schema格式
        schema = {
            "name": name,
            "description": description,
            "properties": parameters.get("properties", {}),
            "required": parameters.get("required", [])
        }
        
        return self.json_schema_to_tool(name, schema, func)
    
    def create_tool_node(self) -> ToolNode:
        """
        创建LangGraph的ToolNode
        
        Returns:
            可在LangGraph工作流中使用的ToolNode
        """
        return ToolNode(self.tools)
    
    def _get_type_from_schema(self, prop_details: Dict) -> Type:
        """
        从Schema定义中获取Python类型
        
        Args:
            prop_details: 属性详情
            
        Returns:
            对应的Python类型
        """
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": List,
            "object": Dict
        }
        
        type_name = prop_details.get("type")
        if not type_name:
            return Any
        
        if type_name in type_map:
            if type_name == "array":
                # 处理数组类型
                items = prop_details.get("items", {})
                item_type = self._get_type_from_schema(items)
                return List[item_type]
            return type_map[type_name]
        return Any


class FunctionRegistry:
    """
    函数注册表，用于管理和注册可在LangGraph中调用的函数
    """
    
    def __init__(self):
        self.functions = {}
        self.converter = LangGraphToolConverter()
        
    def register(self, func: Callable = None, *, name: str = None):
        """
        注册函数为工具
        可以作为装饰器使用
        
        Args:
            func: 要注册的函数
            name: 自定义名称，不提供则使用函数名
            
        Returns:
            注册后的函数
        """
        def decorator(f):
            func_name = name or f.__name__
            self.functions[func_name] = f
            return f
            
        if func is None:
            return decorator
        return decorator(func)
    
    def create_tool_node(self) -> ToolNode:
        """
        从所有注册的函数创建工具节点
        
        Returns:
            LangGraph工具节点
        """
        tools = []
        for func in self.functions.values():
            tool_func = self.converter.function_to_tool(func)
            tools.append(tool_func)
        
        return ToolNode(tools)
    
    def get_tools(self) -> List[BaseTool]:
        """
        获取所有工具的列表
        
        Returns:
            工具列表
        """
        tools = []
        for func in self.functions.values():
            tool_func = self.converter.function_to_tool(func)
            tools.append(tool_func)
        return tools


# 示例用法
if __name__ == "__main__":
    # 示例1：直接使用LangGraphToolConverter
    converter = LangGraphToolConverter()
    
    def add(a: int, b: int) -> int:
        """将两个数字相加"""
        return a + b
    
    # 转换为工具
    add_tool = converter.function_to_tool(add)
    
    # 创建工具节点
    tool_node = converter.create_tool_node()
    
    # 示例2：使用函数注册表
    registry = FunctionRegistry()
    
    @registry.register
    def multiply(a: int, b: int) -> int:
        """将两个数字相乘"""
        return a * b
    
    @registry.register(name="divide_numbers")
    def divide(a: int, b: int) -> float:
        """将两个数字相除"""
        return a / b
    
    # 获取工具节点
    tools_node = registry.create_tool_node()