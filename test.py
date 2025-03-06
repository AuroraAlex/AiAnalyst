"""
这是一个Langgraph节点示例，演示如何使用Langgraph创建一个工作流程来运行你的Agent
包括让大模型自主选择工具的功能
"""

import os
from pathlib import Path
from typing import TypedDict, Annotated, Sequence, Dict, Any, Literal, List, Optional
from agents.image_analysis import ImageAnalysisAgent
from agents.indicators_analysis import IndicatorsAnalysisAgent
from tools.utils import load_config

# 导入Langgraph相关库
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# 导入LangChain相关库，用于实现工具自选功能
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool, BaseTool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from pydantic import BaseModel, Field

# 定义状态类型
class AgentState(TypedDict):
    query: str
    image_path: str
    response: str
    agent_type: str
    intermediate_steps: list

# 定义工具的参数模型
class ImageAnalysisInput(BaseModel):
    query: str = Field(description="分析图像的查询或问题")
    image_path: str = Field(description="要分析的图像文件路径")

class IndicatorsAnalysisInput(BaseModel):
    query: str = Field(description="关于股票指标或技术分析的查询")

# 初始化agents
def init_agents():
    # 加载配置
    config_dir = Path(__file__).parent / "config"
    api_keys = load_config(config_dir / "api_keys.json")
    model_config = load_config(config_dir / "models.json")
    
    # 初始化agents
    image_agent = ImageAnalysisAgent(config={**api_keys, **model_config})
    # 初始化 image_tool
    image_agent.initialize()
    
    indicators_agent = IndicatorsAnalysisAgent(config={**api_keys, **model_config})
    
    return {
        "image": image_agent,
        "indicators": indicators_agent,
        "api_keys": api_keys,
        "model_config": model_config
    }

# 创建一个图像分析工具
def create_image_analysis_tool(image_agent: ImageAnalysisAgent) -> BaseTool:
    def analyze_image_func(query: str, image_path: str) -> str:
        """分析图像并返回结果。需要提供具体的查询问题和图像路径。"""
        try:
            print(f"调用图像分析工具 - 查询：'{query}'，图片路径：'{image_path}'")
            return image_agent.chat(query, image_path)
        except Exception as e:
            return f"图像分析错误：{str(e)}"
    
    return Tool.from_function(
        name="analyze_image",
        description="当用户需要分析图像时使用此工具。需要提供图像路径和具体的分析需求。",
        func=analyze_image_func,
        args_schema=ImageAnalysisInput
    )

# 创建一个指标分析工具
def create_indicators_analysis_tool(indicators_agent: IndicatorsAnalysisAgent) -> BaseTool:
    def analyze_indicators_func(query: str) -> str:
        """分析股票指标并返回结果。需要提供具体的指标分析问题。"""
        try:
            print(f"调用指标分析工具 - 查询：'{query}'")
            # 由于原始Agent没有chat方法，这里返回模拟响应
            return f"指标分析结果：{query}（这是示例响应，实际实现需要根据IndicatorsAnalysisAgent的API）"
        except Exception as e:
            return f"指标分析错误：{str(e)}"
    
    return Tool.from_function(
        name="analyze_indicators",
        description="当用户需要分析股票指标、技术趋势或市场数据时使用此工具。需要提供具体的指标分析问题。",
        func=analyze_indicators_func,
        args_schema=IndicatorsAnalysisInput
    )

# 创建一个自动选择工具的Agent
def create_tool_chooser_agent(api_keys: Dict, model_config: Dict):
    """创建一个能够自主选择工具的Agent"""
    # 修复：正确获取API密钥和模型配置路径
    bailian_api_key = api_keys.get("api_keys", {}).get("bailian_api_key")
    indicators_model_config = model_config.get("models", {}).get("indicators_analysis", {})
    base_url = indicators_model_config.get("base_url")
    model_name = indicators_model_config.get("model_name", "qwen-plus")
    
    print(f"使用模型: {model_name}")
    print(f"使用API基础URL: {base_url}")
    # 输出API密钥的前5个字符，用于调试
    if bailian_api_key:
        print(f"API密钥前缀: {bailian_api_key[:5]}...")
    else:
        print("警告: API密钥为空!")
    
    model = ChatOpenAI(
        model_name=model_name,
        openai_api_key=bailian_api_key,
        openai_api_base=base_url,
        temperature=0
    )
    
    # 初始化agents
    agents = init_agents()
    image_agent = agents["image"]
    indicators_agent = agents["indicators"]
    
    # 创建工具列表
    tools = [
        create_image_analysis_tool(image_agent),
        create_indicators_analysis_tool(indicators_agent)
    ]
    
    # 将工具转换为OpenAI函数格式
    functions = [format_tool_to_openai_function(t) for t in tools]
    
    # 创建Agent提示模板
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""你是一个智能分析助手，专门处理金融和图像分析任务。
        根据用户的输入，你需要选择适当的工具来完成任务。
        
        你有两个工具可以使用：
        
        1. analyze_image - 用于分析图像。使用此工具时，你需要两个参数：
           - query: 用户对图像的具体分析问题
           - image_path: 图像文件的路径
        
        2. analyze_indicators - 用于分析股票指标。使用此工具时，你需要一个参数：
           - query: 用户对股票指标的具体分析问题
        
        当用户提到图像分析或提供了图像路径，请使用analyze_image工具。
        当用户提到股票指标、技术分析或市场趋势，请使用analyze_indicators工具。
        
        你必须从用户的输入中提取关键信息来填充工具所需的参数，并调用适当的工具。
        用户输入可能会明确提及图像路径，需要将该路径作为analyze_image工具的参数。"""),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # 创建OpenAI函数Agent
    agent = create_openai_functions_agent(
        llm=model,
        tools=tools,
        prompt=prompt
    )
    
    # 创建AgentExecutor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

# 定义节点函数
def route_query(state: AgentState) -> Dict[str, Any]:
    """根据查询内容决定使用哪个agent"""
    query = state["query"].lower()
    if "图" in query or "image" in query or "picture" in query:
        # 返回下一个节点的字典，而不是字符串
        return {"agent_type": "image_agent"}
    else:
        return {"agent_type": "indicators_agent"}

def process_image_agent(state: AgentState) -> Dict[str, Any]:
    """处理图像相关查询"""
    agents = init_agents()
    image_agent = agents["image"]
    
    # 如果提供了图像路径，则使用它
    if state.get("image_path"):
        response = image_agent.chat(state["query"], state["image_path"])
    else:
        response = image_agent.chat(state["query"])
    
    return {"response": response}

def process_indicators_agent(state: AgentState) -> Dict[str, Any]:
    """处理指标相关查询"""
    agents = init_agents()
    indicators_agent = agents["indicators"]
    
    # 由于IndicatorsAnalysisAgent没有chat方法，我们模拟一个简单的响应
    response = f"收到指标分析请求：{state['query']}。这里返回一个示例响应，请实现IndicatorsAnalysisAgent的方法。"
    
    return {"response": response}

def process_tool_chooser_agent(state: AgentState) -> Dict[str, Any]:
    """让大模型自己选择合适的工具来处理查询"""
    # 获取所有必要的agents和配置
    agents_and_configs = init_agents()
    api_keys = agents_and_configs["api_keys"]
    model_config = agents_and_configs["model_config"]
    
    # 创建自主选择工具的Agent
    agent_executor = create_tool_chooser_agent(api_keys, model_config)
    
    # 准备输入 - 使用更明确的提示，包含所有必要信息
    query = state["query"]
    image_path = state.get("image_path", "")
    
    if image_path:
        # 确保图像路径被清晰地描述出来
        if "图片" in query or "图像" in query or "image" in query:
            input_text = f"{query}。请使用图像分析工具，图片路径是 '{image_path}'。"
        else:
            input_text = f"请分析图片上的{query}。图片路径是 '{image_path}'。"
    else:
        input_text = query
    
    input_data = {
        "input": input_text,
        "chat_history": []
    }
    
    print(f"\n发送给Agent的查询: '{input_text}'")
    
    # 执行Agent
    result = agent_executor.invoke(input_data)
    
    # 返回结果和中间步骤
    return {
        "response": result["output"],
        "intermediate_steps": result["intermediate_steps"]
    }

# 创建基本工作流图
def create_agent_graph():
    # 创建图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("router", route_query)
    workflow.add_node("image_agent", process_image_agent)
    workflow.add_node("indicators_agent", process_indicators_agent)
    
    # 根据agent_type路由到相应处理函数
    def router_condition(state: AgentState) -> Literal["image_agent", "indicators_agent"]:
        return state["agent_type"]
    
    # 添加条件边
    workflow.add_conditional_edges(
        "router",
        router_condition,
        {
            "image_agent": "image_agent", 
            "indicators_agent": "indicators_agent"
        }
    )
    
    # 设置入口点
    workflow.set_entry_point("router")
    
    # 编译工作流
    return workflow.compile()

# 创建使用工具选择Agent的工作流图
def create_tool_chooser_graph():
    # 创建图
    workflow = StateGraph(AgentState)
    
    # 添加工具选择Agent节点
    workflow.add_node("tool_chooser", process_tool_chooser_agent)
    
    # 设置入口点
    workflow.set_entry_point("tool_chooser")
    
    # 编译工作流
    return workflow.compile()

# 运行示例
if __name__ == "__main__":
    # 选择要运行的工作流类型
    use_tool_chooser = True
    
    if use_tool_chooser:
        # 使用自动工具选择的Agent
        print("正在使用自动工具选择Agent...")
        agent_graph = create_tool_chooser_graph()
    else:
        # 使用基本的路由图
        print("正在使用基本路由图...")
        agent_graph = create_agent_graph()
    
    # 创建一个内存检查点保存器
    memory_saver = MemorySaver()
    
    # 测试查询
    print("正在测试Langgraph工作流...")
    
    # 示例1: 图像分析
    result1 = agent_graph.invoke(
        {
            "query": "分析这张图片上的股票趋势走向如何？",
            "image_path": "./coca_cola_stock.png",
            "agent_type": "",
            "response": "",
            "intermediate_steps": []
        },
        config={"checkpointer": memory_saver}
    )
    print("\n图像分析结果:", result1["response"])
    if "intermediate_steps" in result1 and result1["intermediate_steps"]:
        print("\n中间步骤:")
        for step in result1["intermediate_steps"]:
            print(f"- 工具: {step[0].tool}")
            print(f"- 输入: {step[0].tool_input}")
            print(f"- 输出: {step[1]}")
    
    # 示例2: 指标分析
    result2 = agent_graph.invoke(
        {
            "query": "请分析MACD和RSI指标的交叉信号对股价的预测意义",
            "image_path": "",
            "agent_type": "",
            "response": "",
            "intermediate_steps": []
        },
        config={"checkpointer": memory_saver}
    )
    print("\n指标分析结果:", result2["response"])
    if "intermediate_steps" in result2 and result2["intermediate_steps"]:
        print("\n中间步骤:")
        for step in result2["intermediate_steps"]:
            print(f"- 工具: {step[0].tool}")
            print(f"- 输入: {step[0].tool_input}")
            print(f"- 输出: {step[1]}")