"""
RAG检索模块 - 基于多模态向量库的统一检索接口
"""
import os
from typing import List, Dict, Any, Optional
from multimodal_vector_store import DualIndexVectorStore, LongTextMultiModalEmbedder


class RAGRetriever:
    """RAG检索器，提供统一的文本和图片查询接口"""

    def __init__(
        self,
        chroma_path: str = "/root/autodl-tmp/ocr_image/chroma_db",
        text_collection_name: str = "text_index",
        image_collection_name: str = "image_index"
    ):
        """
        初始化RAG检索器

        Args:
            chroma_path: ChromaDB持久化目录路径
            text_collection_name: 文本索引集合名称
            image_collection_name: 图片索引集合名称
        """
        self.embedder = LongTextMultiModalEmbedder()
        self.vector_store = DualIndexVectorStore(
            persist_directory=chroma_path,
            text_collection_name=text_collection_name,
            image_collection_name=image_collection_name,
            embedder=self.embedder
        )

    def retrieve_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        text_weight: float = 0.7,
        image_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        通过文本查询检索相关文档

        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            text_weight: 文本索引权重
            image_weight: 图片索引权重

        Returns:
            检索结果列表，按相关性排序
        """
        if not query_text or not query_text.strip():
            return []

        results = self.vector_store.search(
            query_text=query_text,
            query_image_path="",
            top_k=top_k,
            text_weight=text_weight,
            image_weight=image_weight
        )
        return results

    def retrieve_by_image(
        self,
        image_path: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        通过图片查询检索相关文档

        Args:
            image_path: 查询图片路径
            top_k: 返回结果数量

        Returns:
            检索结果列表，按相关性排序
        """
        if not image_path or not os.path.exists(image_path):
            return []

        results = self.vector_store.search(
            query_text="",
            query_image_path=image_path,
            top_k=top_k
        )
        return results

    def retrieve_hybrid(
        self,
        query_text: str = "",
        image_path: str = "",
        top_k: int = 5,
        text_weight: float = 0.7,
        image_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        混合查询（文本+图片）检索相关文档

        Args:
            query_text: 查询文本（可选）
            image_path: 查询图片路径（可选）
            top_k: 返回结果数量
            text_weight: 文本权重（当有文本查询时）
            image_weight: 图片权重（当有文本查询时）

        Returns:
            检索结果列表，按相关性排序
        """
        results = self.vector_store.search(
            query_text=query_text,
            query_image_path=image_path,
            top_k=top_k,
            text_weight=text_weight,
            image_weight=image_weight
        )
        return results


# 简单的测试函数
def test_retriever():
    """测试检索器功能"""
    retriever = RAGRetriever()

    # 测试文本查询
    print("测试文本查询...")
    text_results = retriever.retrieve_by_text("我想更换健身追踪器的表带，有其他尺寸可选吗？", top_k=3)
    print(f"找到 {len(text_results)} 个相关结果")
    for i, result in enumerate(text_results[:2]):
        print(f"结果 {i+1}: {result['doc'][:100]}...")
        print(f"得分: {result['score']:.4f}")
        print(f"元数据: {result['meta']}")
        print("-" * 50)

    # 如果有测试图片，可以测试图片查询
    test_image_path = "/root/autodl-tmp/ocr_image/images_standard/air_conditioner_01.png"
    if os.path.exists(test_image_path):
        print("\n测试图片查询...")
        image_results = retriever.retrieve_by_image(test_image_path, top_k=3)
        print(f"找到 {len(image_results)} 个相关结果")


if __name__ == "__main__":
    test_retriever()