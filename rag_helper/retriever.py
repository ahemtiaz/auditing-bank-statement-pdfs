from __future__ import annotations

from typing import Dict, List, Literal, Tuple, Callable, Any
import faiss
import numpy as np

# Safe torch import to handle DLL loading routine failures dynamically on Windows
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

from sentence_transformers import SentenceTransformer

from abc import ABC, abstractmethod
import ir_measures
from ir_measures import *

from pathlib import Path
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Automatically download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class DocumentProvider:
    """Load chunk corpus and expose views and helpers."""

    def __init__(self, 
                 csv_path: str | Path, 
                 use_nltk_preprocessor: bool = True
    ) -> None:
        self.csv_path = csv_path
        df = pd.read_csv(csv_path, usecols=["chunk_id", "text_description", "image_filename", "query"])
        self._ids: List[str] = df["chunk_id"].tolist()
        self._texts: List[str] = df["text_description"].fillna("").astype(str).tolist()
        self._queries: List[str] = [q for q in df["query"].fillna("").astype(str).tolist() if q.strip()]
        self._chunk_to_page: Dict[str, int] = dict(zip(df["chunk_id"], df["image_filename"]))
        self._tokens: list[list[str]] | None = None
        self._embed_cache: Dict[Any, List[Any]] = {}

        # Flag to switch between simple .split() and your NLTK-based preprocessing
        self._use_nltk_preprocessor = use_nltk_preprocessor

    # basic access -----------------------------------------------------------
    @property
    def ids(self) -> List[str]:
        return self._ids

    @property
    def tokens(self) -> List[List[str]]:
        return self._token_view
    
    @property
    def texts(self) -> List[str]:
        return self._texts

    @property
    def chunk_to_page(self) -> Dict[str, int]:
        return self._chunk_to_page

    @property
    def stats(self) -> Dict[str, int]:
        """Return statistics about the document collection:
        - unique_pages: number of unique pages (image filenames)
        - total_chunks: total number of chunks
        - unique_queries: number of unique queries
        """
        return {
            "unique_pages": len(set(self._chunk_to_page.values())),
            "total_chunks": len(set(self._ids)),  # Only unique chunk_ids
            "unique_queries": len(set(self._queries))
        }

    # internal views ---------------------------------------------------------
    @property
    def _token_view(self) -> List[List[str]]:
        if self._tokens is None:
            if self._use_nltk_preprocessor:
                # wrap list of texts into a dict for preprocess_text
                passages = {i: txt for i, txt in enumerate(self._texts)}
                self._tokens = self.preprocess_text(passages)
            else:
                self._tokens = [t.split() for t in self._texts]
        return self._tokens
    
    def preprocess_text(self, passages: Dict[int, str]) -> List[List[str]]:
        """
        Preprocess text by performing the following steps:
            - Remove stopwords
            - Strip punctuation (retain only alphanumeric tokens)
            - Convert all words to lowercase
        """
        stop_words = set(stopwords.words("english"))
        tokenized_list = [
            [
                word.lower()
                for word in word_tokenize(sentence)
                if word.isalnum() and word.lower() not in stop_words
            ]
            for sentence in passages.values()
        ]
        return tokenized_list

    def _embedding_view(self, encode_fn: Callable[[List[str]], List[Any]]) -> List[Any]:
        if encode_fn not in self._embed_cache:
            self._embed_cache[encode_fn] = encode_fn(self._texts)
        return self._embed_cache[encode_fn]

    # main getter ------------------------------------------------------------
    def get(
        self,
        kind: str = "text",
        *,
        encode_fn: Callable[[List[str]], List[Any]] | None = None,
    ) -> Tuple[List[str], List[Any]]:
        if kind in {"text", "raw"}:
            return self._ids, self._texts
        if kind in {"tokens", "bm25"}:
            return self._ids, self._token_view
        if kind == "dense":
            if encode_fn is None:
                raise ValueError("encode_fn required for dense view")
            return self._ids, self._embedding_view(encode_fn)
        raise ValueError("unknown kind")


class Evaluator_ir:
    """Light wrapper around *ir_measures* that prints and returns cut-off metrics, MRR, and R-Precision."""

    def __init__(self) -> None:
        pass
        
    def evaluate(
        self,
        run: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        k_values: List[int],
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate retrieval *run* against *qrels* using *ir_measures*."""
        print("Evaluating run with ir_measures...")
        
        measures = []
        for k in k_values:
            measures += [nDCG@k, P@k, Recall@k, AP@k]
        # global metrics
        measures += [RR, Rprec]
       
        results = ir_measures.calc_aggregate(measures, qrels, run)

        # Organize by k and global
        metrics_by_k: Dict[str, Dict[str, float]] = {}
        for k in k_values:
            metrics_by_k[k] = {
                'ndcg': results.get(nDCG@k, 0.0),
                'precision': results.get(P@k, 0.0),
                'recall': results.get(Recall@k, 0.0)
            }
        metrics_by_k['global'] = {
            'mrr': results.get(RR, 0.0),
            'rprec': results.get(Rprec, 0.0)
        }

        # Pretty print
        if verbose:
            print("\n=== Evaluation Results ===")
            for k in k_values:
                m = metrics_by_k[k]
                print(
                    f"K={k:<2}  NDCG:{m['ndcg']:.4f}  P:{m['precision']:.4f}  R:{m['recall']:.4f}"
                )
            g = metrics_by_k['global']
            print(f"GLOBAL MRR:{g['mrr']:.4f}  Rprec:{g['rprec']:.4f}")

        return metrics_by_k


class BaseRetriever(ABC):
    def __init__(self):
        self.evaluator_ir = Evaluator_ir()
    
    @abstractmethod
    def search(self, queries: Dict[str, str], **kwargs) -> Dict[str, Dict[str, float]]:
        """Search method to be implemented by child classes"""
        pass

    def evaluate(self, run: Dict[str, Dict[str, float]], 
                qrels: Dict[str, Dict[str, int]], 
                k_values: list = [1, 3, 5, 10],
                verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Evaluate retrieval results
        Args:
            run: Dict[qid -> Dict[doc_id -> score]]
            qrels: Dict[qid -> Dict[doc_id -> relevance]]
            verbose: Whether to print results
        Returns:
            Evaluation metrics for different k values
        """
        return self.evaluator_ir.evaluate(run, qrels, k_values, verbose)


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def download_retriever_models() -> None:
    """Download the required RAG models into the local models directory if they are not already present."""
    from huggingface_hub import snapshot_download
    
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    required_models = {
        "BAAI/bge-m3": "bge-m3",
        "sentence-transformers/all-MiniLM-L6-v2": "all-MiniLM-L6-v2"
    }
    
    for repo_id, folder_name in required_models.items():
        local_path = models_dir / folder_name
        # Check if model folder exists and is valid (contains config.json)
        if local_path.exists() and (local_path / "config.json").exists():
            print(f"[OK] Model '{repo_id}' already exists locally at '{local_path.resolve()}'.")
        else:
            print(f"Downloading model '{repo_id}' to local folder '{local_path.resolve()}'...")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_path,
                    local_dir_use_symlinks=False
                )
                print(f"[OK] Successfully downloaded '{repo_id}' to '{local_path.resolve()}'.")
            except Exception as e:
                print(f"[ERROR] Failed to download model '{repo_id}': {e}")
                raise e


class SentenceTransformerRetriever(BaseRetriever):
    """Dense retriever using SentenceTransformer + FAISS (full-corpus scoring)."""

    def __init__(
        self,
        provider: DocumentProvider,
        model_name: str = "BAAI/bge-m3",
        device_map: str = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu",
        is_instruct: bool = False,
        task_description: str = "Given a user query, retrieve the most relevant passages from the document corpus",
    ) -> None:
        super().__init__()
        self.is_instruct = is_instruct
        self.task_description = task_description

        # Resolve model name/path dynamically to prevent hardcoded paths and support offline local load
        resolved_model_path = model_name
        if not Path(model_name).exists():
            model_basename = model_name.split("/")[-1]
            local_path = Path(__file__).parent / "models" / model_basename
            if local_path.exists() and (local_path / "config.json").exists():
                resolved_model_path = str(local_path.resolve())
                print(f"Resolved model '{model_name}' to local path: {resolved_model_path}")
            else:
                print(f"Warning: Local model path '{local_path}' not found or incomplete. Falling back to hub load.")

        # Check for cached embeddings
        csv_path = getattr(provider, "csv_path", None)
        cache_path = None
        if csv_path:
            clean_model_name = str(model_name).replace("/", "_").replace("\\", "_").replace(":", "_")
            cache_path = Path(csv_path).parent / f"{Path(csv_path).stem}_{clean_model_name}_embeddings.npy"

        if cache_path and cache_path.exists():
            print(f"Loading cached embeddings from {cache_path} ...")
            embeds = np.load(cache_path)
            self.model = SentenceTransformer(resolved_model_path, device=device_map)
            ids = provider.ids
        else:
            self.model = SentenceTransformer(resolved_model_path, device=device_map)
            print("Encoding corpus (this may take a few minutes on CPU)...")
            ids, embeds = provider.get("dense", encode_fn=self._encode)
            if cache_path:
                print(f"Caching embeddings to {cache_path} ...")
                np.save(cache_path, embeds)


        self.doc_ids: List[str] = ids
        self.chunk_to_page = provider.chunk_to_page  # map chunk_id → page_number

        self.doc_embeddings = np.asarray(embeds, dtype="float32")
        faiss.normalize_L2(self.doc_embeddings)
        dim = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.doc_embeddings)

    # ------------------------------------------------------------------
    def _encode(self, texts: List[str]) -> List[List[float]]:  # used by provider cache
        return (
            self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            .astype("float32")
            .tolist()
        )
    # ------------------------------------------------------------------

    def _aggregate_scores(self, vals: list[float], agg: Literal["max", "mean", "sum"]) -> float:
        if agg == "max":
            return float(max(vals))
        elif agg == "sum":
            return float(sum(vals))
        elif agg == "mean":
            return float(np.mean(vals))
        else:
            raise ValueError("Unsupported aggregation method.")
        
    def search(self, queries: Dict[str, str], agg: Literal["max", "mean", "sum"] = "max") -> Dict[str, Dict[str, float]]:
        """Score every chunk, then use max score per page."""
        run: Dict[str, Dict[str, float]] = {}
        docs = self.doc_embeddings

        for qid, qtext in queries.items():
            if self.is_instruct:
                q_input = get_detailed_instruct(self.task_description, qtext)
            else:
                q_input = qtext

            q_emb = self.model.encode([q_input], normalize_embeddings=True, show_progress_bar=False).astype("float32")
            scores = (docs @ q_emb[0]).tolist()

            page_scores: Dict[str, List[float]] = {}
            for doc_id, sc in zip(self.doc_ids, scores):
                val = self.chunk_to_page.get(doc_id)
                if val is None or not isinstance(val, str):
                    continue
                pg = val.split('.')[0]  # remove .png extension
                page_scores.setdefault(str(pg), []).append(sc)

            run[qid] = {p: self._aggregate_scores(vals, agg) for p, vals in page_scores.items()}
        return run
