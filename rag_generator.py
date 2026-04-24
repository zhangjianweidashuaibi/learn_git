"""
RAG生成模块 - 基于检索结果生成答案（支持阿里云百炼平台）
"""
import os
import json
import requests
from typing import List, Dict, Any, Optional
from rag_retriever import RAGRetriever


class RAGGenerator:
    """RAG生成器，基于检索结果和大语言模型生成答案"""

    def __init__(self, retriever: RAGRetriever):
        """
        初始化RAG生成器

        Args:
            retriever: RAG检索器实例
        """
        self.retriever = retriever

        # 阿里云百炼平台配置
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.model_name = os.getenv("LLM_MODEL_NAME", "qwen-max")  # 可选: qwen-max, qwen-plus, qwen-turbo等

        if not self.api_key:
            print("警告: 未设置DASHSCOPE_API_KEY环境变量")
            print("请在autodl服务器上设置: export DASHSCOPE_API_KEY='your-dashscope-api-key'")
            print("暂时使用模拟模式返回检索结果")

    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        格式化检索到的上下文

        Args:
            results: 检索结果列表

        Returns:
            格式化的上下文字符串
        """
        if not results:
            return "未找到相关信息。"

        context_parts = []
        for i, result in enumerate(results, 1):
            doc_text = result['doc']
            # 提取实际的文本内容（去除前面的"文本:"前缀）
            if doc_text.startswith("文本:"):
                actual_text = doc_text[3:doc_text.find("...") if "..." in doc_text else len(doc_text)]
            else:
                actual_text = doc_text

            context_parts.append(f"文档 {i}: {actual_text}")

        return "\n\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """
        构建LLM提示词

        Args:
            query: 用户查询
            context: 检索到的上下文

        Returns:
            完整的提示词
        """
        prompt = f"""你是一个专业的技术支持助手，请基于以下提供的上下文信息回答用户的问题。

上下文信息：
{context}

用户问题：{query}

请根据上下文信息提供准确、简洁的回答。如果上下文信息不足以回答问题，请说明无法提供相关信息。
"""
        return prompt

    def generate_answer(
        self,
        query: str,
        top_k: int = 3,
        text_weight: float = 0.9,
        image_weight: float = 0.1
    ) -> str:
        """
        生成问答答案

        Args:
            query: 用户查询
            top_k: 检索结果数量
            text_weight: 文本权重
            image_weight: 图片权重

        Returns:
            生成的答案
        """
        # 1. 检索相关文档
        results = self.retriever.retrieve_by_text(
            query_text=query,
            top_k=top_k,
            text_weight=text_weight,
            image_weight=image_weight
        )

        # 2. 格式化上下文
        context = self._format_context(results)

        # 3. 构建提示词
        prompt = self._build_prompt(query, context)

        # 4. 调用LLM生成答案
        if not self.api_key:
            return f"[模拟模式 - 未设置DASHSCOPE_API_KEY]\n基于以下上下文生成答案:\n\n问题: {query}\n\n上下文:\n{context}\n\n提示: 请设置DASHSCOPE_API_KEY环境变量以启用真实API调用"
        else:
            return self._call_dashscope_api(prompt)

    def _call_dashscope_api(self, prompt: str) -> str:
        """
        调用阿里云百炼平台API生成答案

        Args:
            prompt: 完整的提示词

        Returns:
            LLM生成的答案
        """
        try:
            url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model_name,
                "input": {
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                },
                "parameters": {
                    "max_tokens": 1000,
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "enable_thinking": False
                }
            }

            # 发送API请求，设置合理的超时时间
            response = requests.post(url, headers=headers, json=data, timeout=30)

            # 检查HTTP状态码
            if response.status_code == 200:
                try:
                    result = response.json()
                    # 确保响应结构正确
                    if "output" in result and "text" in result["output"]:
                        return result["output"]["text"]
                    else:
                        return f"API响应格式异常: 缺少output.text字段"
                except json.JSONDecodeError:
                    return f"API响应解析失败: 无法解析JSON响应"
            else:
                # 尝试解析错误信息
                try:
                    error_data = response.json()
                    error_msg = error_data.get("message", error_data.get("code", f"HTTP {response.status_code}"))
                except:
                    error_msg = f"HTTP {response.status_code}"
                return f"调用阿里云百炼API时出错: {error_msg}"

        except requests.exceptions.Timeout:
            return "调用阿里云百炼API超时，请检查网络连接或增加超时时间"
        except requests.exceptions.ConnectionError:
            return "无法连接到阿里云百炼API，请检查网络连接"
        except requests.exceptions.RequestException as e:
            return f"HTTP请求异常: {str(e)}"
        except Exception as e:
            return f"调用阿里云百炼API时发生未知错误: {str(e)}"


# 简单的测试函数
def test_generator():
    """测试生成器功能"""
    retriever = RAGRetriever()
    generator = RAGGenerator(retriever)

    # 测试问答
    query = "我想更换健身追踪器的表带，有其他尺寸可选吗？"
    print(f"用户问题: {query}")
    print("\n生成答案:")
    answer = generator.generate_answer(query, top_k=3)
    print(answer)


if __name__ == "__main__":
    test_generator()