"""
支持长文本的多模态RAG向量化方案
核心改进：
1. 长文本使用支持长文档的模型
2. 图片使用CLIP编码
3. 双重索引+检索融合
"""
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

# 🔥 新增：服务器图片根目录（核心配置）
SERVER_IMAGE_DIR = "/root/autodl-tmp/ocr_image/images_standard"

# 🔥 新增：自动把Windows路径转换为Linux服务器路径
def convert_windows_path_to_linux(windows_path: str) -> str:
    if not windows_path:
        return ""
    # 提取图片文件名（Manual27_13.png）
    img_name = os.path.basename(windows_path)
    # 拼接Linux路径
    return os.path.join(SERVER_IMAGE_DIR, img_name)

class LongTextMultiModalEmbedder:
    """支持长文本的多模态嵌入器（Python3.8 完美兼容）"""

    def __init__(
        self,
        text_model_name: str = "BAAI/bge-m3",
        image_model_name: str = "clip-ViT-B-32"
    ):
        print(f"加载文本模型: {text_model_name}...")
        self.text_model = SentenceTransformer(
            text_model_name,
            cache_folder="./models_cache"
        )
        self.text_model.max_seq_length = 8192
        self.text_dim = self.text_model.get_sentence_embedding_dimension()
        print(f"文本向量维度: {self.text_dim}")

        print(f"加载图片模型: {image_model_name}...")
        self.image_model = SentenceTransformer(
            image_model_name,
            cache_folder="./models_cache"
        )
        self.image_dim = self.image_model.get_sentence_embedding_dimension() or 512
        print(f"图片向量维度: {self.image_dim}")

    def embed_text(self, text: str) -> np.ndarray:
        """文本独立向量化"""
        if not text or text.strip() == "":
            return np.zeros(self.text_dim)
        return self.text_model.encode(
            text, 
            normalize_embeddings=True,
            max_length=8192,
            truncation=True
        )

    def embed_image(self, image_path: str) -> np.ndarray:
        """图片独立向量化"""
        try:
            # 🔥 关键修改：自动转换路径！！
            image_path = convert_windows_path_to_linux(image_path)
            image = Image.open(image_path).convert("RGB")
            return self.image_model.encode(image, normalize_embeddings=True)
        except Exception as e:
            print(f"警告: 无法加载图片 {image_path}: {e}")
            return np.zeros(self.image_dim)

    def embed_text_for_cross_modal(self, text: str) -> np.ndarray:
        """跨模态文本编码（CLIP）"""
        if not text or text.strip() == "":
            return np.zeros(self.image_dim)
        return self.image_model.encode(text, normalize_embeddings=True)

    def embed_mixed_content(
        self,
        text: str,
        image_paths: List[str],
        ocr_texts: List[List[str]] = None,
        visual_descriptions: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        enhanced_text = text
        if ocr_texts:
            ocr_content = " ".join([" ".join(ocr) for ocr in ocr_texts])
            enhanced_text = f"{enhanced_text}\nOCR: {ocr_content}"
        if visual_descriptions:
            enhanced_text = f"{enhanced_text}\n图片描述: {' '.join(visual_descriptions)}"

        text_emb = self.embed_text(enhanced_text)

        image_emb = np.zeros(self.image_dim)
        if image_paths:
            embeddings = [self.embed_image(p) for p in image_paths if self.embed_image(p).any()]
            if embeddings:
                image_emb = np.mean(embeddings, axis=0)
                image_emb = image_emb / np.linalg.norm(image_emb)

        return text_emb, image_emb

class DualIndexVectorStore:
    """双重索引：文本索引 + 图片索引，独立存储，检索融合"""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        text_collection_name: str = "text_index",
        image_collection_name: str = "image_index",
        embedder: Optional[LongTextMultiModalEmbedder] = None
    ):
        # 初始化客户端
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )

        # 🔥 修复1：安全创建/获取集合（不会重复删除，更稳定）
        self.text_collection = self.client.get_or_create_collection(
            name=text_collection_name, 
            metadata={"hnsw:space": "cosine"}
        )
        self.image_collection = self.client.get_or_create_collection(
            name=image_collection_name, 
            metadata={"hnsw:space": "cosine"}
        )

        self.embedder = embedder or LongTextMultiModalEmbedder()

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        text_data = []
        image_data = []

        for chunk in chunks:
            chunk_id = chunk["id"]
            text_emb, image_emb = self.embedder.embed_mixed_content(
                text=chunk.get("text", ""),
                image_paths=chunk.get("image_paths", []),
                ocr_texts=chunk.get("ocr_texts", []),
                visual_descriptions=chunk.get("visual_descriptions", [])
            )
            doc = f"文本:{chunk.get('text','')[:300]}... 图片数:{len(chunk.get('image_paths',[]))}"
            meta = {
                "chunk_id": chunk_id,
                "has_images": len(chunk.get("image_paths",[]))>0,
                "chunk_type": "mixed" if text_emb.any() and image_emb.any() else "text_only" if text_emb.any() else "image_only"
            }

            if text_emb.any():
                text_data.append((chunk_id, text_emb.tolist(), doc, meta))
            if image_emb.any():
                image_data.append((chunk_id, image_emb.tolist(), doc, meta))

        if text_data:
            ids, embs, docs, metas = zip(*text_data)
            self.text_collection.add(ids=list(ids), embeddings=list(embs), documents=list(docs), metadatas=list(metas))
        if image_data:
            ids, embs, docs, metas = zip(*image_data)
            self.image_collection.add(ids=list(ids), embeddings=list(embs), documents=list(docs), metadatas=list(metas))
        return len(chunks)

    def search(
        self,
        query_text: str = "",
        query_image_path: str = "",
        top_k: int = 5,
        text_weight: float = 0.7,
        image_weight: float = 0.3
    ) -> List[Dict]:
        results = {}

        if query_text:
            emb = self.embedder.embed_text(query_text)
            res = self.text_collection.query(query_embeddings=[emb.tolist()], n_results=top_k)
            self._merge(res, results, text_weight)

            emb_clip = self.embedder.embed_text_for_cross_modal(query_text)
            res_img = self.image_collection.query(query_embeddings=[emb_clip.tolist()], n_results=top_k)
            self._merge(res_img, results, image_weight)

        if query_image_path:
            emb = self.embedder.embed_image(query_image_path)
            res = self.image_collection.query(query_embeddings=[emb.tolist()], n_results=top_k)
            self._merge(res, results, 1.0)

        return sorted(results.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    def _merge(self, query_res, results, weight):
        if not query_res["ids"][0]: return
        for i, cid in enumerate(query_res["ids"][0]):
            score = (1 - query_res["distances"][0][i]) * weight
            if cid not in results:
                results[cid] = {"id": cid, "score": 0, "doc": query_res["documents"][0][i], "meta": query_res["metadatas"][0][i]}
            results[cid]["score"] += score

    def reset_collections(self):
        # 清空集合数据（不删除集合，避免报错）
        self.text_collection.delete()
        self.image_collection.delete()

def load_chunks_from_folder(folder_path: str) -> List[Dict]:
    chunks = []
    for f in Path(folder_path).glob("*.json"):
        with open(f, encoding="utf-8") as fp:
            chunks.extend(json.load(fp))
    return chunks

def build_vector_store(chunk_folder, chroma_path):
    print("构建多模态RAG向量库...")
    embedder = LongTextMultiModalEmbedder()
    store = DualIndexVectorStore(chroma_path, embedder=embedder)
    chunks = load_chunks_from_folder(chunk_folder)
    print(f"加载 {len(chunks)} 个数据块")
    
    # 🔥 修复2：删除这里重复的reset调用（核心问题！）
    # store.reset_collections()  # 这行代码导致集合被清空，直接注释/删除
    
    store.add_chunks(chunks)
    print("向量库构建完成！")
    return store

if __name__ == "__main__":
    vector_store = build_vector_store(
        chunk_folder="/root/autodl-tmp/ocr_image/chunk_results",
        chroma_path="/root/autodl-tmp/ocr_image/chroma_db"
    )