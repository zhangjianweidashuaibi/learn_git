"""
端到端RAG问答系统测试脚本
"""
import os
from rag_retriever import RAGRetriever
from rag_generator import RAGGenerator


def main():
    """主函数 - 端到端RAG问答测试"""

    # 配置向量库路径（根据你的autodl环境）
    CHROMA_PATH = "/root/autodl-tmp/ocr_image/chroma_db"

    print("初始化RAG系统...")
    retriever = RAGRetriever(chroma_path=CHROMA_PATH)
    generator = RAGGenerator(retriever)

    print("RAG系统初始化完成！")
    print("=" * 50)

    # 测试用例（基于健身追踪器手册）
    test_queries = [
        "如何更换健身追踪器的表带？",
        "健身追踪器支持哪些运动模式？",
        "如何给设备充电？",
        "设备的防水等级是多少？",
        "如何重置设备到出厂设置？"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n测试问题 {i}: {query}")
        print("-" * 30)

        try:
            # 先测试检索
            results = retriever.retrieve_by_text(query, top_k=2)
            print(f"检索到 {len(results)} 个相关文档")

            if results:
                print("相关文档预览:")
                for j, result in enumerate(results[:1]):
                    doc_preview = result['doc'][:150] + "..." if len(result['doc']) > 150 else result['doc']
                    print(f"  - {doc_preview}")
                    print(f"    相关性得分: {result['score']:.4f}")

            # 再测试生成答案
            answer = generator.generate_answer(query, top_k=2)
            print(f"\n生成答案:\n{answer}")

        except Exception as e:
            print(f"处理问题时出错: {e}")

        print("=" * 50)

    # 交互式问答循环
    print("\n进入交互式问答模式（输入 'quit' 退出）:")
    while True:
        user_query = input("\n请输入您的问题: ").strip()
        if user_query.lower() in ['quit', 'exit', '退出']:
            break

        if not user_query:
            continue

        try:
            answer = generator.generate_answer(user_query, top_k=3)
            print(f"\n答案:\n{answer}")
        except Exception as e:
            print(f"生成答案时出错: {e}")


if __name__ == "__main__":
    main()