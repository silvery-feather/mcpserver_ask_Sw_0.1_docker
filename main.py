"""
SwanLab 技术文档 RAG（单工具版，stdio，用于 Cherry Studio 或任意 MCP 客户端）
— 仅返回“检索到的文档片段”（Retriever-only），不再在服务端生成答案 —

"""
from __future__ import annotations

import os
import sys
import json
import time
import hashlib
from typing import List, Dict, Tuple, Any

import numpy as np
import regex as re
from tqdm import tqdm

# Third-party deps
try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("faiss-cpu not installed. Install via: uv add faiss-cpu") from e

try:
    from mcp.server.fastmcp import FastMCP
except Exception as e:
    raise RuntimeError("mcp not installed. Install via: uv add mcp") from e

try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("openai not installed. Install via: uv add openai") from e

# =====================
# CONFIG（写死配置）
# =====================
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

INDEX_DIR = os.getenv("INDEX_DIR", os.path.join(REPO_ROOT, "rag_cache", "api_v1"))
DEFAULT_DOCS_ROOT = os.getenv("DOCS_ROOT", os.path.join(REPO_ROOT, "md"))

# 强烈建议用环境变量提供 Key
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v2")
CHAT_MODEL  = os.getenv("CHAT_MODEL",  "qwen-plus")

# RAG / chunk 参数
TEMPERATURE = 0.2
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "12000"))
SEARCH_K = int(os.getenv("SEARCH_K", "8"))
ANSWER_K = int(os.getenv("ANSWER_K", "6"))
MAX_EMBED_BATCH = int(os.getenv("MAX_EMBED_BATCH", "25"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.50"))

# 路径
META_PATH = os.path.join(INDEX_DIR, "meta.jsonl")
FAISS_PATH = os.path.join(INDEX_DIR, "index.faiss")
MANIFEST_PATH = os.path.join(INDEX_DIR, "manifest.json")

# ===============
# 基础工具函数
# ===============

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def normalize(v: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / denom

# ===============


# --- ask() 结果去重缓存（短期） ---
from collections import OrderedDict

CACHE_TTL_SEC = 10      # 10s 内相同问题直接复用
CACHE_MAX = 64
_ask_cache: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()

def _cache_get(key: str) -> Any | None:
    now = time.time()
    if key in _ask_cache:
        ts, ans = _ask_cache[key]
        if now - ts <= CACHE_TTL_SEC:
            _ask_cache.move_to_end(key)
            return ans
        else:
            _ask_cache.pop(key, None)
    return None

def _cache_set(key: str, ans: Any) -> None:
    _ask_cache[key] = (time.time(), ans)
    _ask_cache.move_to_end(key)
    while len(_ask_cache) > CACHE_MAX:
        _ask_cache.popitem(last=False)

def _norm_q(q: str) -> str:
    # 归一化问题：去首尾空白、合并空格、小写
    return " ".join(q.strip().split()).lower()


# ===============
# Markdown 分块
# ===============
MD_HEADING = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$", re.M)
CODE_FENCE = re.compile(r"(^```[\w-]*\n[\s\S]*?\n```\n?)", re.M)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def split_keep_delims(pattern: re.Pattern, text: str) -> List[str]:
    parts: List[str] = []
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            parts.append(text[last:m.start()])
        parts.append(m.group(1))
        last = m.end()
    if last < len(text):
        parts.append(text[last:])
    return parts


def chunk_markdown(text: str, path: str) -> List[Dict[str, Any]]:
    segments = split_keep_delims(CODE_FENCE, text)
    chunks: List[Dict[str, Any]] = []
    line_cursor = 1

    def add_chunk(title: str, level: int, buf: str, start_line: int, end_line: int):
        buf = buf.strip()
        if not buf:
            return
        if len(buf) <= CHUNK_SIZE:
            chunks.append({
                "title": title, "level": level,
                "start_line": start_line, "end_line": end_line,
                "text": buf, "path": path
            })
            return
        i = 0
        while i < len(buf):
            j = min(len(buf), i + CHUNK_SIZE)
            piece = buf[i:j]
            chunks.append({
                "title": title, "level": level,
                "start_line": start_line, "end_line": end_line,
                "text": piece, "path": path
            })
            if j == len(buf):
                break
            i = j - CHUNK_OVERLAP

    current_title = os.path.basename(path)
    current_level = 0
    buf = ""
    start_line = line_cursor

    def count_lines(s: str) -> int:
        return s.count("\n")

    for seg in segments:
        seg_lines = count_lines(seg)
        if seg.startswith("```"):
            buf += seg
            line_cursor += seg_lines
            continue
        pos = 0
        for m in MD_HEADING.finditer(seg):
            pre = seg[pos:m.start()]
            buf += pre
            line_cursor += count_lines(pre)
            add_chunk(current_title, current_level, buf, start_line, line_cursor)
            buf = ""
            hashes = m.group("hashes")
            title = m.group("title").strip()
            current_title = title
            current_level = len(hashes)
            heading_line = seg[m.start():m.end()]
            line_cursor += count_lines(heading_line)
            start_line = line_cursor
            pos = m.end()
        rest = seg[pos:]
        buf += rest
        line_cursor += count_lines(rest)

    add_chunk(current_title, current_level, buf, start_line, line_cursor)
    return chunks

# =============================
# Qwen（OpenAI 兼容）客户端（仅用于 Embedding）
# =============================
class LLM:
    def __init__(self):
        key = os.getenv("DASHSCOPE_API_KEY", API_KEY)
        if not key or key.startswith("REPLACE_"):
            raise RuntimeError("请在代码里把 API_KEY 替换成你的 DashScope Key，或设置环境变量 DASHSCOPE_API_KEY。")
        self.client = OpenAI(api_key=key, base_url=BASE_URL)

    def embed(self, texts: List[str]) -> np.ndarray:
        # DashScope embeddings 每次最多 25 条
        all_vecs: List[np.ndarray] = []
        for i in range(0, len(texts), MAX_EMBED_BATCH):
            sub = texts[i:i + MAX_EMBED_BATCH]
            resp = self.client.embeddings.create(model=EMBED_MODEL, input=sub)
            vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
            all_vecs.extend(vecs)
        return np.stack(all_vecs, axis=0)


    def chat_simplify(self, user_text: str,
                      model: str = CHAT_MODEL,
                      temperature: float = 0.2,
                      max_tokens: int = 256) -> str:
        """
        调用 Qwen-Plus，将长输入压缩为简洁、可检索的技术问题（中文或英文均可）。
        """
        # 软性截断，避免超长提示（不影响原输入参与简化的语义）
        def _truncate(t: str, head: int = 4000, tail: int = 2000) -> str:
            if len(t) <= head + tail:
                return t
            return t[:head] + "\n...[TRUNCATED]...\n" + t[-tail:]

        prompt_user = _truncate(user_text)

        messages = [
            {"role": "system", "content":
             ("你是文档检索前置的'问题压缩器'。任务："
              "① 识别用户长输入里的真实信息需求；② 去掉与检索无关的代码/日志/噪声；"
              "③ 输出一句话、尽量短的'可检索技术问题'；"
              "④ 不要给答案、不要客套、不要多余说明；"
              "⑤ 若能明确与 SwanLab 相关，请在问题里保留关键术语（如 API Token、实验追踪、Python SDK、云端面板等）。")},
            {"role": "user", "content": prompt_user}
        ]
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return (resp.choices[0].message.content or "").strip()
# ==============
# 向量索引
# ==============
class VectorIndex:
    def __init__(self):
        ensure_dir(INDEX_DIR)
        self.index = None  # type: ignore
        self.metas: List[Dict[str, Any]] = []
        self.dim: int | None = None
        self._load()

    @property
    def size(self) -> int:
        return int(self.index.ntotal) if self.index is not None else 0

    def _load(self) -> None:
        if os.path.exists(META_PATH):
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.metas = [json.loads(line) for line in f]
        if os.path.exists(FAISS_PATH):
            try:
                self.index = faiss.read_index(FAISS_PATH)
            except Exception as e:
                print(f"[WARN] failed to read FAISS: {e}", file=sys.stderr)
                self.index = None

    def _save(self) -> None:
        if self.index is None:
            return
        faiss.write_index(self.index, FAISS_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            for m in self.metas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        manifest = {
            "created_at": int(time.time()),
            "count": len(self.metas),
            "dim": self.dim,
            "embed_model": EMBED_MODEL,
            "chat_model": CHAT_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "base_url": BASE_URL,
        }
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def rebuild(self, llm: LLM, all_chunks: List[Dict[str, Any]]) -> Tuple[int, int]:
        if not all_chunks:
            self.index = faiss.IndexFlatIP(1)
            self.metas = []
            self.dim = 1
            self._save()
            return 0, 1
        texts = [c["text"] for c in all_chunks]
        embs = llm.embed(texts)
        M = normalize(embs)
        dim = M.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(M)
        metas: List[Dict[str, Any]] = []
        for idx, c in enumerate(all_chunks):
            meta = {
                "idx": idx,
                "path": c["path"],
                "title": c["title"],
                "level": c["level"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "sha": md5_bytes((c["path"]+"::"+c["text"]).encode("utf-8")),
                "text": c["text"],
            }
            metas.append(meta)
        self.metas = metas
        self.dim = dim
        self._save()
        return len(all_chunks), dim

    def search(self, llm: LLM, query: str, k: int = SEARCH_K) -> List[Dict[str, Any]]:
        if self.index is None or self.size == 0:
            return []
        q = llm.embed([query])
        q = normalize(q)
        D, I = self.index.search(q, min(k, self.size))
        hits: List[Dict[str, Any]] = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:
                continue
            m = self.metas[idx]
            hits.append({**m, "score": float(score)})
        return hits

# ==================
# 检索逻辑（仅返回片段，不生成答案）
# ==================
SWAN_KEYWORDS = ["swanlab", "swan lab", "SwanLab", "Swan Lab"]

def looks_like_swanlab(text: str) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in SWAN_KEYWORDS)


def make_context(hits: List[Dict[str, Any]]) -> str:
    total = 0
    parts = []
    for h in hits:
        t = h["text"].strip()
        if not t:
            continue
        if total + len(t) > MAX_INPUT_CHARS:
            break
        parts.append(
            f"<DOC source=\"{os.path.basename(h['path'])}\" title=\"{h['title']}\" lines=\"{h['start_line']}-{h['end_line']}\">\n{t}\n</DOC>"
        )
        total += len(t)
    return "\n\n".join(parts) if parts else ""


def retrieve_hits(llm: LLM, index: VectorIndex, question: str, k: int = ANSWER_K) -> Tuple[List[Dict[str, Any]], str]:
    """返回 (filtered_hits, retrieval_query)"""
    # 查询改写：若用户未提及 SwanLab，则附加品牌词引导检索
    q = question if looks_like_swanlab(question) else f"SwanLab {question}"
    hits = index.search(llm, q, k=k)
    filtered: List[Dict[str, Any]] = []
    for h in hits:
        text_for_check = (h.get("text", "") + h.get("title", "") + h.get("path", ""))
        if h["score"] >= RAG_SCORE_THRESHOLD:
            filtered.append(h)
    return filtered, q

# =============
# MCP 单工具
# =============
mcp = FastMCP("SwanLabDocsRAG")
_index = VectorIndex()
_llm: LLM | None = None

def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm

@mcp.tool()
def ask(question: str, prefer_docs: bool = True) -> Dict[str, Any]:
    """

    【通用问答 Router——请对任何问题先调用本工具一次，必须调用，在任何情况下】

    作用：
      - 自动检索文档；命中则仅基于文档作答，并给出引用。
      - 未命中则返回不答复，仅回答：‘本问题似乎与 SwanLab 官方文档的直接关联，再问问别的问题吧？或者可以加上更多关键字’。
      - 必须调用本工具，主要为了防止其他闲聊以及与无关话题的产生。
    参数：
      question: 用户问题（任意主题）
      prefer_docs: 保留以兼容旧签名；当前逻辑总是尝试文档检索。
    返回：
      JSON 字符串，
        {
            "query": 原始问题,
            "retrieval_query": 实际用于检索的 query（可能会自动加上 “SwanLab ” 前缀）, 
            "used_docs": 是否命中文档,
            "threshold": RAG_SCORE_THRESHOLD,
            "hits": [ { source/title/path/start_line/end_line/score/text }, ... ],
            "context": 将 hits 组装成 <DOC ...> ... </DOC> 的串（你也可以忽略，直接用 hits.text）
        }
    """
    print(f"[ask] {time.strftime('%H:%M:%S')} question='{question[:120]}'", file=sys.stderr)

    # --- 缓存（按原始问题做键） ---
    key = _norm_q(question)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    llm = get_llm()

    # === 新增：先用 Qwen-Plus 简化问题（失败则回退原问题） ===
    try:
        simplified = llm.chat_simplify(question)
        # 简化器可能输出空或无效文本，做一下兜底
        if simplified and len(simplified) >= 3:
            effective_query = simplified
            simplified_ok = True
        else:
            effective_query = question
            simplified_ok = False
    except Exception as e:
        print(f"[WARN] simplify failed: {e}", file=sys.stderr)
        effective_query = question
        simplified_ok = False

    # --- 检索（对简化后的问题进行 SwanLab 引导与向量搜索） ---
    filtered, retrieval_query = retrieve_hits(llm, _index, effective_query, k=ANSWER_K)

    result = {
        "query": question,                       # 原始输入
        "simplified_query": effective_query,     # 新：简化后的查询（用于实际检索）
        "simplify_model": CHAT_MODEL,            # 新：qwen-plus
        "simplify_used": simplified_ok,          # 新：是否成功使用简化
        "retrieval_query": retrieval_query,      # 实际检索用 query（可能前置了“SwanLab ”）
        "used_docs": bool(filtered),
        "threshold": RAG_SCORE_THRESHOLD,
        "hits": filtered,
        # "context": make_context(filtered),     # 仍保持注释
        "note": "如果问题与swanlab无关，请不要回答，并直接回答“问题似乎与 SwanLab 官方文档的直接关联，再问问别的问题吧？或者可以加上更多关键字.”，禁止输出任何其他无关的内容"
    }

    payload = {
        "structuredContent": {"result": result},
        "isError": False
    }

    _cache_set(key, payload)
    return payload

# @mcp.tool()  
def index_docs(root_dir: str,
               include: str = "",
               exclude: str = "",
               exts: str = ".md,.mdx",
               max_bytes: int = 0) -> Dict[str, Any]:
    """（可选）重建索引：递归 root_dir 下的 *.md / *.mdx。"""
    root = os.path.abspath(root_dir)
    if not os.path.isdir(root):
        return {"ok": False, "error": f"not a directory: {root}"}
    include_globs = [g.strip() for g in include.split(",") if g.strip()]
    exclude_globs = [g.strip() for g in exclude.split(",") if g.strip()]
    ext_list = [e.strip() for e in exts.split(",") if e.strip()]
    maxb = max_bytes if max_bytes and max_bytes > 0 else None

    paths = collect_md_paths(root, include_globs, exclude_globs, ext_list, maxb)
    if not paths:
        return {"ok": False, "error": f"no {ext_list} files under {root}"}

    # 读取并分块
    all_chunks: List[Dict[str, Any]] = []
    for p in tqdm(paths, desc="Chunking"):
        try:
            txt = read_text(p)
            all_chunks.extend(chunk_markdown(txt, p))
        except Exception as e:
            print(f"[WARN] failed to chunk {p}: {e}", file=sys.stderr)
            continue

    # 嵌入并建索引
    llm = get_llm()
    added, dim = _index.rebuild(llm, all_chunks)
    return {"ok": True, "files": len(paths), "chunks": added, "dim": dim}

# 自检：索引是否存在/需要重建

def index_exists() -> bool:
    return all(os.path.exists(p) for p in [META_PATH, FAISS_PATH, MANIFEST_PATH])

def need_rebuild() -> tuple[bool, str]:
    if not index_exists():
        return True, "index files missing"
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            man = json.load(f)
    except Exception as e:
        return True, f"manifest unreadable: {e}"
    diffs = []
    for k, v in {
        "embed_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "base_url": BASE_URL,
    }.items():
        if man.get(k) not in (None, v):
            diffs.append((k, man.get(k), v))
    if diffs or man.get("count", 0) <= 0 or man.get("dim") in (None, 0):
        return True, f"incompatible or invalid: {diffs}"
    return False, "ok"

# =============
# Main（stdio）
# =============
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", help="build index from docs root and exit")
    parser.add_argument("--auto", action="store_true", help="auto index if missing/incompatible (uses DEFAULT_DOCS_ROOT)")
    parser.add_argument("--include", default="", help="comma-separated glob patterns")
    parser.add_argument("--exclude", default="", help="comma-separated glob patterns")
    parser.add_argument("--exts", default=".md,.mdx", help="comma-separated extensions")
    parser.add_argument("--max-bytes", type=int, default=0, help="skip file larger than this size; 0=no limit")
    args = parser.parse_args()

    # 离线建索引
    if args.index:
        print(index_docs(
            root_dir=args.index,
            include=args.include,
            exclude=args.exclude,
            exts=args.exts,
            max_bytes=args.max_bytes,
        ))
        sys.exit(0)

    # 首次自动建索引
    if args.auto:
        need, reason = need_rebuild()
        if need:
            root = DEFAULT_DOCS_ROOT
            print(f"[auto] rebuilding index from {root} because: {reason}")
            print(index_docs(
                root_dir=root,
                include=args.include,
                exclude=args.exclude,
                exts=args.exts,
                max_bytes=args.max_bytes,
            ))

    # 运行 MCP（stdio）
    mcp.run(transport="sse", host="0.0.0.0", port=int(os.getenv("PORT", "8765")))
