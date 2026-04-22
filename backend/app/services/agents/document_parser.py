"""
法规文档结构化解析器 (P3 M4)

提供 RegulationChunk 层级模型和层级感知 PDF 解析器，
替代纯 CharacterTextSplitter 的扁平切分。

层级结构：
  Document → Chapter → Section → Paragraph → Chunk

每个 RegulationChunk 包含：
  - hierarchy_path: 层级路径（如 "Chapter 2 > Section 2.1 > Paragraph 2.1.3"）
  - parent_id: 父级 chunk 的 ID
  - children_ids: 子级 chunk 的 ID 列表
  - cross_references: 交叉引用列表（先留空，后续迭代填充）
  - section_title: 章节标题
"""
import hashlib
import re
from typing import Dict, List, Optional

from langchain_core.documents import Document
from pydantic import BaseModel, Field


# ==========================================
# RegulationChunk 层级模型
# ==========================================

class RegulationChunk(BaseModel):
    """法规文档结构化 Chunk 模型

    包含层级元数据，支持父子关系和交叉引用。
    """

    chunk_id: str = Field(
        ..., description="唯一标识（基于内容的哈希）",
    )
    page_content: str = Field(
        ..., min_length=10, description="Chunk 文本内容",
    )
    hierarchy_path: str = Field(
        "", description="层级路径，如 'Chapter 2 > Section 2.1 > Paragraph 2.1.3'",
    )
    hierarchy_level: int = Field(
        0, ge=0, description="层级深度（0=document, 1=chapter, 2=section, 3=paragraph）",
    )
    section_title: str = Field(
        "", description="章节标题文本",
    )
    parent_id: Optional[str] = Field(
        None, description="父级 chunk 的 ID（顶层为 None）",
    )
    children_ids: List[str] = Field(
        default_factory=list, description="子级 chunk 的 ID 列表",
    )
    cross_references: List[str] = Field(
        default_factory=list, description="交叉引用的 chunk ID 列表（预留，当前为空）",
    )
    page_number: int = Field(
        -1, ge=-1, description="来源页码",
    )
    source_document: str = Field(
        "", description="来源文档名",
    )

    def to_document(self) -> Document:
        """转换为 LangChain Document（含完整元数据）

        ChromaDB 不接受空列表或 None 作为 metadata 值，因此过滤掉。
        """
        raw_meta = {
            "chunk_id": self.chunk_id,
            "hierarchy_path": self.hierarchy_path,
            "hierarchy_level": self.hierarchy_level,
            "section_title": self.section_title,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "cross_references": self.cross_references,
            "page": self.page_number,
            "source_document": self.source_document,
        }
        # ChromaDB 兼容：过滤空列表和 None 值
        clean_meta = {
            k: v for k, v in raw_meta.items()
            if v is not None and v != [] and v != ""
        }
        return Document(
            page_content=self.page_content,
            metadata=clean_meta,
        )

    @staticmethod
    def compute_chunk_id(content: str, hierarchy_path: str) -> str:
        """计算 chunk 唯一 ID"""
        normalized = re.sub(r'\s+', ' ', (content + hierarchy_path).strip().lower())
        return hashlib.md5(normalized.encode()).hexdigest()[:12]


# ==========================================
# 法规层级标题正则模式
# ==========================================

# 按层级深度排列（从高到低）
HIERARCHY_PATTERNS = [
    # Level 1: Chapter / Part / Schedule
    (1, r'^#{0,3}\s*(Chapter|Part|Schedule)\s+([\d.]+)\s*:?\s*(.*)'),
    # Level 2: Section
    (2, r'^#{0,3}\s*Section\s+([\d.]+)\s*:?\s*(.*)'),
    # Level 3: Paragraph / Clause
    (3, r'^#{0,3}\s*(Paragraph|Clause)\s+([\d.]+)\s*:?\s*(.*)'),
    # Level 2 alternative: 数字编号如 "2.1 Title"
    (2, r'^#{0,3}\s*(\d+\.\d+)\s+([A-Z][^\n]*)'),
    # Level 1 alternative: 纯数字编号如 "1 Title"
    (1, r'^#{0,3}\s*(\d+)\.\s+([A-Z][^\n]*)'),
    # Appendix
    (1, r'^#{0,3}\s*Appendix\s+([A-Z])\s*:?\s*(.*)'),
]


def _classify_heading(line: str) -> Optional[tuple]:
    """判断一行文本是否是法规层级标题

    Returns:
        (level, title_text) 或 None
    """
    stripped = line.strip()
    if not stripped:
        return None

    for level, pattern in HIERARCHY_PATTERNS:
        match = re.match(pattern, stripped, re.IGNORECASE)
        if match:
            groups = match.groups()
            # 构建标题文本
            if len(groups) >= 3:
                title_text = f"{groups[0]} {groups[1]}: {groups[2]}".strip(": ")
            elif len(groups) == 2:
                title_text = f"{groups[0]} {groups[1]}".strip(": ")
            else:
                title_text = groups[0]
            return (level, title_text)

    return None


def _build_hierarchy_path(parent_stack: List[tuple]) -> str:
    """从父级栈构建层级路径字符串"""
    parts = [title for _, title, _ in parent_stack]
    return " > ".join(parts) if parts else ""


def parse_pdf_with_hierarchy(
    pages: List[Document],
    source_name: str = "",
    min_section_length: int = 50,
) -> List[RegulationChunk]:
    """层级感知 PDF 解析器

    将 PDF 页面按法规文档的层级结构（Chapter > Section > Paragraph）
    切分为 RegulationChunk，并建立父子关系。

    Args:
        pages: PyPDFLoader 加载的原始页面列表
        source_name: 来源文档名称
        min_section_length: 最小段落长度（低于此值的段落合并到父级）

    Returns:
        RegulationChunk 列表（含完整层级元数据）
    """
    all_chunks: List[RegulationChunk] = []
    # 父级栈: [(level, title, chunk_id)]
    parent_stack: List[tuple] = []
    # chunk_id → RegulationChunk 的映射（用于建立父子关系）
    chunk_map: Dict[str, RegulationChunk] = {}

    for page in pages:
        page_num = page.metadata.get("page", -1)
        lines = page.page_content.split('\n')

        current_lines: List[str] = []
        current_level = 0
        current_title = "Preamble"

        for line in lines:
            heading_info = _classify_heading(line)

            if heading_info is not None:
                new_level, new_title = heading_info

                # 保存当前段落到 chunk
                if current_lines:
                    _flush_section(
                        lines=current_lines,
                        level=current_level,
                        title=current_title,
                        page_num=page_num,
                        source_name=source_name,
                        parent_stack=parent_stack,
                        all_chunks=all_chunks,
                        chunk_map=chunk_map,
                        min_section_length=min_section_length,
                    )

                # 更新父级栈
                # 弹出所有 >= new_level 的父级
                while parent_stack and parent_stack[-1][0] >= new_level:
                    parent_stack.pop()

                parent_stack.append((new_level, new_title, None))  # chunk_id 稍后填充
                current_level = new_level
                current_title = new_title
                current_lines = [line]
            else:
                current_lines.append(line)

        # 页面末尾的段落
        if current_lines:
            _flush_section(
                lines=current_lines,
                level=current_level,
                title=current_title,
                page_num=page_num,
                source_name=source_name,
                parent_stack=parent_stack,
                all_chunks=all_chunks,
                chunk_map=chunk_map,
                min_section_length=min_section_length,
            )

    # 第二遍：更新 parent_stack 中暂存的 chunk_id
    _resolve_parent_ids(all_chunks, chunk_map)

    return all_chunks


def _flush_section(
    lines: List[str],
    level: int,
    title: str,
    page_num: int,
    source_name: str,
    parent_stack: List[tuple],
    all_chunks: List[RegulationChunk],
    chunk_map: Dict[str, RegulationChunk],
    min_section_length: int,
) -> None:
    """将当前段落的行列表刷新为一个 RegulationChunk"""
    content = '\n'.join(lines).strip()
    if len(content) < min_section_length:
        return

    hierarchy_path = _build_hierarchy_path(parent_stack)
    chunk_id = RegulationChunk.compute_chunk_id(content, hierarchy_path)

    # 确定 parent_id
    parent_id = None
    if len(parent_stack) >= 2:
        # 父级是栈中倒数第二个元素（当前标题刚被 push）
        parent_level, parent_title, _ = parent_stack[-2]
        parent_path = _build_hierarchy_path(parent_stack[:-1])
        parent_id = RegulationChunk.compute_chunk_id("", parent_path)

    chunk = RegulationChunk(
        chunk_id=chunk_id,
        page_content=content,
        hierarchy_path=hierarchy_path or title,
        hierarchy_level=level,
        section_title=title,
        parent_id=parent_id,
        page_number=page_num,
        source_document=source_name,
    )

    all_chunks.append(chunk)
    chunk_map[chunk_id] = chunk

    # 更新 parent_stack 中当前层的 chunk_id
    if parent_stack:
        last = parent_stack[-1]
        parent_stack[-1] = (last[0], last[1], chunk_id)


def _resolve_parent_ids(
    all_chunks: List[RegulationChunk],
    chunk_map: Dict[str, RegulationChunk],
) -> None:
    """第二遍扫描：补充父子关系

    - 为每个 chunk 查找其 parent_id 对应的父 chunk
    - 将自身 ID 添加到父 chunk 的 children_ids
    """
    for chunk in all_chunks:
        if chunk.parent_id and chunk.parent_id in chunk_map:
            parent = chunk_map[chunk.parent_id]
            if chunk.chunk_id not in parent.children_ids:
                parent.children_ids.append(chunk.chunk_id)


def regulation_chunks_to_documents(chunks: List[RegulationChunk]) -> List[Document]:
    """将 RegulationChunk 列表转换为 LangChain Document 列表"""
    return [chunk.to_document() for chunk in chunks]
