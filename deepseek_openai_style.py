import os
import json
import base64
import logging
from typing import Optional, Dict, Any
from openai import OpenAI
from pathlib import Path
from datetime import datetime


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """配置相关错误"""
    pass


class ImageProcessingError(Exception):
    """图片处理相关错误"""
    pass


class ConfigManager:
    """配置管理类"""
    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self.config = self._load_config()
        logger.info(f"配置加载成功: {self.config_path}")
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"配置文件不存在: {self.config_path}")
            raise ConfigError(f"配置文件 {self.config_path} 不存在")
        except json.JSONDecodeError:
            logger.error(f"配置文件格式错误: {self.config_path}")
            raise ConfigError(f"配置文件 {self.config_path} 格式不正确")
    
    @property
    def api_key(self) -> str:
        return self.config['api_key']
    
    @property
    def image_folder(self) -> str:
        return self.config['image_folder']
    
    @property
    def output_folder(self) -> str:
        return self.config['output_folder']
    
    @property
    def base_url(self) -> str:
        return self.config['base_url']
    
    @property
    def model_config(self) -> Dict[str, str]:
        return self.config['model_config']

    @property
    def save_json(self) -> bool:
        """是否保存JSON数据"""
        return self.config.get('save_json', False)  # 默认为False


class ImageProcessor:
    """图片处理类"""
    def __init__(self, config: ConfigManager):
        self.config = config
        self._ensure_folders()
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.conversation_history = []  # 存储对话历史
        self.current_image_base64 = None  # 存储当前处理的图片
        self.current_output_file = None  # 存储当前输出文件
        logger.info("图片处理器初始化完成")

    def _ensure_folders(self) -> None:
        """确保必要的文件夹存在"""
        try:
            Path(self.config.image_folder).mkdir(exist_ok=True)
            Path(self.config.output_folder).mkdir(exist_ok=True)
            logger.debug("确保必要的文件夹存在")
        except Exception as e:
            logger.error(f"创建文件夹失败: {str(e)}")
            raise ImageProcessingError(f"创建文件夹失败: {str(e)}")

    def _encode_image(self, image_path: str) -> str:
        """Base64编码图片"""
        try:
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
                logger.debug(f"图片编码成功: {image_path}")
                return encoded
        except Exception as e:
            logger.error(f"图片编码失败: {str(e)}")
            raise ImageProcessingError(f"图片编码失败: {str(e)}")

    def _get_output_filename(self, image_name: str) -> str:
        """生成输出文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(image_name).stem
        return f"{base_name}_{timestamp}.txt"

    def process_external_image(self, image_path: str, prompt_text: str = "图中描绘的是什么景象？") -> None:
        """
        处理任意位置的图片并获取AI响应，会将图片复制到images文件夹
        :param image_path: 图片的完整路径
        :param prompt_text: 提问文本
        """
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"图片文件不存在: {image_path}")
            raise FileNotFoundError(f"图片文件 {image_path} 不存在")
        
        try:
            logger.info(f"开始处理外部图片: {image_path}")
            
            # 复制图片到images文件夹
            target_path = Path(self.config.image_folder) / image_path.name
            import shutil
            shutil.copy2(image_path, target_path)
            logger.info(f"已复制图片到: {target_path}")
            
            # 处理图片
            self.current_image_base64 = self._encode_image(str(target_path))
            self.conversation_history = []  # 重置对话历史
            completion = self._create_completion(self.current_image_base64, prompt_text)
            
            # 创建输出文件
            self.current_output_file = Path(self.config.output_folder) / self._get_output_filename(image_path.name)
            self._process_completion(completion, self.current_output_file)
            
            logger.info(f"图片处理完成，结果已保存至: {self.current_output_file}")
        except Exception as e:
            logger.error(f"处理图片时出错: {str(e)}")
            raise ImageProcessingError(f"处理图片失败: {str(e)}")

    def process_image(self, image_name: str, prompt_text: str = "图中描绘的是什么景象？") -> None:
        """
        处理指定图片并获取AI响应
        :param image_name: 图片文件名（位于images文件夹中）
        :param prompt_text: 提问文本
        """
        image_path = Path(self.config.image_folder) / image_name
        if not image_path.exists():
            logger.error(f"图片文件不存在: {image_path}")
            raise FileNotFoundError(f"图片文件 {image_path} 不存在")
        
        try:
            logger.info(f"开始处理图片: {image_name}")
            self.current_image_base64 = self._encode_image(str(image_path))
            self.conversation_history = []  # 重置对话历史
            completion = self._create_completion(self.current_image_base64, prompt_text)
            
            # 创建输出文件
            self.current_output_file = Path(self.config.output_folder) / self._get_output_filename(image_name)
            self._process_completion(completion, self.current_output_file)
            
            logger.info(f"图片处理完成，结果已保存至: {self.current_output_file}")
        except Exception as e:
            logger.error(f"处理图片时出错: {str(e)}")
            raise ImageProcessingError(f"处理图片失败: {str(e)}")

    def continue_conversation(self, prompt_text: str) -> None:
        """
        继续当前图片的对话
        :param prompt_text: 新的提问文本
        """
        if not self.current_image_base64 or not self.current_output_file:
            logger.error("没有正在进行的对话")
            raise ImageProcessingError("请先使用process_image开始一个新的对话")
        
        try:
            logger.info("继续对话...")
            completion = self._create_completion(self.current_image_base64, prompt_text)
            self._process_completion(completion, self.current_output_file)
            logger.info(f"对话继续，结果已追加至: {self.current_output_file}")
        except Exception as e:
            logger.error(f"继续对话时出错: {str(e)}")
            raise ImageProcessingError(f"继续对话失败: {str(e)}")

    def _create_completion(self, base64_image: str, prompt_text: str):
        """创建AI完成请求"""
        try:
            # 构建消息历史
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                }
            ]
            
            # 如果是新对话，添加图片
            if not self.conversation_history:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                        {"type": "text", "text": prompt_text}
                    ]
                })
            else:
                # 添加历史对话记录
                messages.extend(self.conversation_history)
                # 添加新的用户提问
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": prompt_text}]
                })

            completion = self.client.chat.completions.create(
                model=self.config.model_config['model'],
                messages=messages,
                modalities=["text"],
                stream=True
            )

            # 保存这次的用户提问到历史记录
            self.conversation_history.append({
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}]
            })
            
            return completion

        except Exception as e:
            logger.error(f"创建AI请求失败: {str(e)}")
            raise ImageProcessingError(f"创建AI请求失败: {str(e)}")

    def _process_completion(self, completion, output_file: Path) -> None:
        """处理AI响应并保存到文件"""
        try:
            # 用于收集所有的内容
            full_content = ""
            json_contents = []
            assistant_response = ""  # 收集助手的完整回复

            # 收集所有chunks
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end='')  # 实时打印到控制台
                    full_content += content
                    assistant_response += content
                    if self.config.save_json:
                        json_contents.append(chunk.model_dump())  # 仅在需要时保存JSON

            # 保存助手的回复到对话历史
            self.conversation_history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_response}]
            })

            # 确定是否需要写入文件头部信息
            is_new_conversation = len(self.conversation_history) <= 2  # 只有系统提示和一轮对话时是新对话

            if is_new_conversation:
                with open(output_file, 'w', encoding='utf-8') as f:
                    # 写入基本信息
                    f.write("=" * 50 + "\n")
                    f.write("AI图像分析结果\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # 写入详细信息
                    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"模型: {self.config.model_config['model']}\n")
                    f.write("-" * 50 + "\n\n")
                    
                    # 写入第一轮对话内容
                    f.write("对话记录:\n\n")
                    f.write(f"问: {self.conversation_history[1]['content'][0]['text']}\n")  # 用户问题
                    f.write(f"答: {assistant_response}\n")
                    f.write("\n" + "-" * 50 + "\n")
                    
                    # 仅在配置为True时写入JSON数据
                    if self.config.save_json:
                        f.write("\n原始JSON数据:\n")
                        f.write("=" * 50 + "\n")
                        for json_content in json_contents:
                            f.write(json.dumps(json_content, ensure_ascii=False, indent=2))
                            f.write("\n")
                        f.write("=" * 50 + "\n\n")
            else:
                # 追加新的对话内容
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n问: {self.conversation_history[-2]['content'][0]['text']}\n")  # 用户问题
                    f.write(f"答: {assistant_response}\n")
                    f.write("\n" + "-" * 50 + "\n")
                    
                    # 仅在配置为True时写入JSON数据
                    if self.config.save_json:
                        f.write("\n原始JSON数据:\n")
                        f.write("=" * 50 + "\n")
                        for json_content in json_contents:
                            f.write(json.dumps(json_content, ensure_ascii=False, indent=2))
                            f.write("\n")
                        f.write("=" * 50 + "\n\n")
            
            # 更新文件末尾
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write("-" * 50 + "\n")
                f.write(f"对话轮次: {len(self.conversation_history) // 2}\n")
                f.write("=" * 50)
                
        except Exception as e:
            logger.error(f"处理AI响应时出错: {str(e)}")
            raise ImageProcessingError(f"处理AI响应失败: {str(e)}")


def main():
    try:
        config = ConfigManager()
        processor = ImageProcessor(config)
        
        # 第一轮对话
        processor.process_image("可口可乐股票.png", "这张图片展示了什么内容？")
        
        # 继续对话
        processor.continue_conversation("请详细分析一下图中的财务数据。")
        processor.continue_conversation("这些数据说明了什么？")
        
    except (ConfigError, ImageProcessingError) as e:
        logger.error(f"程序运行错误: {str(e)}")
    except Exception as e:
        logger.critical(f"未预期的错误: {str(e)}")


if __name__ == "__main__":
    main()