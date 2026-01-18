

from openai import OpenAI
from tinyrag.llm.base_llm import BaseLLM
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoConfig 
import os
os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'
import warnings
warnings.filterwarnings("ignore")
class qwen3_llm(BaseLLM):
    def __init__(self, model_id_key, device = "cpu", is_api=False):
        super().__init__(model_id_key, device, is_api)
        load_dotenv()
        self.api_key = os.getenv("LLM_API_KEY")
        self.model_id = os.getenv("LLM_MODEL_ID")
        self.url = os.getenv("LLM_BASE_URL")

        self.client = OpenAI(
            base_url=self.url,  # ModelScope API端点
            api_key=self.api_key  # 您的ModelScope token
        )
         # 只需要加载tokenizer用于文本处理
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_key,  # 使用合适的Qwen3模型标识
            trust_remote_code=True,
            cache_dir="./models"
        )
        # 移除本地模型加载相关代码
        self.model = None  # 不再需要本地模型实例
         # 设置模型配置
        self.config = {
            "model": model_id_key,  # 使用传入的模型标识
            "max_tokens": 2048,
            "temperature": 0
        }
    def generate(self, content: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]

        try:
            # 关键修改：添加 stream=True 参数
            response = self.client.chat.completions.create(
                model=self.model_id_key,
                messages=messages,
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                stream=True,  # 启用流式传输 需要配套使用
                extra_body={
                    "enable_thinking": True  # 启用思考模式
                }
            )

            # 处理流式响应
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content

            return full_response

        except Exception as e:
            print(f"API调用失败: {e}")
            return f"生成失败: {str(e)}" 
           
def main():
    llm = qwen3_llm("Qwen/Qwen3-8B")
    print(llm.generate("今天天气怎样？"))
if __name__=="__main__":
    main()