"""Gene IO functions (Phase 7.11c extracted from gene_codec.py)

Contains:
  - save_gene: save gene to JSON file
  - load_gene: load gene from JSON file
  - genes_to_jsonl: save gene list to JSONL
  - genes_from_jsonl: load gene list from JSONL
"""
from __future__ import annotations

import json
import logging
from typing import List

logger = logging.getLogger(__name__)

from .gene_codec import AgentGene


def save_gene(gene: AgentGene, filepath: str) -> None:
    """保存基因到JSON文件"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(gene.to_dict(), f, ensure_ascii=False, indent=2, default=str)


def load_gene(filepath: str) -> AgentGene:
    """从JSON文件加载基因"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return AgentGene.from_dict(data)


def genes_to_jsonl(genes: List[AgentGene], filepath: str) -> None:
    """将基因列表保存为JSONL文件（每行一个基因）"""
    with open(filepath, "w", encoding="utf-8") as f:
        for gene in genes:
            f.write(json.dumps(gene.to_dict(), ensure_ascii=False, default=str) + "\n")


def genes_from_jsonl(filepath: str) -> List[AgentGene]:
    """从JSONL文件加载基因列表"""
    genes = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                genes.append(AgentGene.from_dict(json.loads(line)))
    return genes
