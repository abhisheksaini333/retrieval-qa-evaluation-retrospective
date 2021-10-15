"""Real CPU Transformer inference with content-checked, offline reloadable bundles."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModel, AutoModelForQuestionAnswering, AutoTokenizer, pipeline

from .core import canonical_hash, corpus_hash, validate_request
from .bundle_io import sha256_file, directory_fingerprint, directory_files as directory_files

ENCODER_ID = "sentence-transformers/paraphrase-MiniLM-L6-v2"
ENCODER_REVISION = "c9a2bfebc254878aee8c3aca9e6844d5bbb102d1"
READER_ID = "distilbert/distilbert-base-cased-distilled-squad"
READER_REVISION = "564e9b582944a57a3e586bbb98fd6f0a4118db7f"




class Encoder:
    def __init__(self, path, threads=2, config=None):
        from .config import EncoderConfig
        self.config = config if config is not None else EncoderConfig()
        if not isinstance(self.config, EncoderConfig):
            raise ValueError("encoder config must be EncoderConfig")
        self.path = Path(path)
        from .runtime import validate_threads
        self.threads = validate_threads(threads)
        from .bundle_io import model_inventory
        model_inventory(self.path)
        self.fingerprint = directory_fingerprint(self.path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                                         clean_up_tokenization_spaces=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                              use_safetensors=True).cpu().eval()
        self.dimension = self.model.config.hidden_size
        if self.config.max_tokens > self.model.config.max_position_embeddings:
            raise ValueError("encoder token limit exceeds model positions")

    def encode(self, texts):
        from .config import validate_encoder_texts
        texts = validate_encoder_texts(texts)
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        batches = []
        for start in range(0, len(texts), self.config.batch_size):
            encoded = self.tokenizer(texts[start:start + self.config.batch_size], padding=True, truncation=True,
                                     max_length=self.config.max_tokens, return_tensors="pt")
            from .runtime import cpu_threads
            with cpu_threads(self.threads), torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden[:, 0] if self.config.pooling == "cls" else
                          (hidden * mask).sum(1) / mask.sum(1).clamp(min=1))
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            batches.append(normalized.numpy())
        return np.concatenate(batches).astype(np.float32)


class DenseIndex:
    @property
    def fingerprint(self):
        return canonical_hash({"corpus": corpus_hash(self.documents), "encoder": self.encoder.fingerprint,
                               "config": self.encoder.config.fingerprint,
                               "vectors": hashlib.sha256(self.embeddings.tobytes()).hexdigest()})

    def __init__(self, documents, encoder, embeddings=None):
        self.documents, self.encoder = list(documents), encoder
        if len({d.doc_id for d in self.documents}) != len(self.documents):
            raise ValueError("duplicate document ID")
        self.embeddings = (encoder.encode([d.title + " " + d.text for d in self.documents])
                           if embeddings is None else embeddings)
        from .config import validate_embeddings
        self.embeddings = validate_embeddings(self.embeddings, len(self.documents), encoder.dimension)

    def search(self, question, k=5):
        validate_request(question, k)
        if not self.documents:
            return []
        similarities = self.embeddings @ self.encoder.encode([question])[0]
        ranked = [(document.doc_id, float(score)) for document, score in zip(self.documents, similarities)]
        from .retrieval import stable_top_k
        return stable_top_k(ranked, k)


    def search_batch(self, questions, k=5):
        from .config import validate_encoder_texts
        from .retrieval import stable_top_k
        questions = validate_encoder_texts(questions)
        for question in questions:
            validate_request(question, k)
        if not self.documents:
            return [[] for _ in questions]
        encoded = self.encoder.encode(questions)
        return [stable_top_k(((document.doc_id, float(score))
                              for document, score in zip(self.documents, self.embeddings @ vector)), k)
                for vector in encoded]

    def search_diverse(self, question, k=5, relevance_weight=0.5):
        from .retrieval import mmr_select
        validate_request(question, k)
        if not self.documents:
            return []
        return mmr_select(self.encoder.encode([question])[0], self.embeddings,
                          [d.doc_id for d in self.documents], k, relevance_weight)


class Reader:
    def __init__(self, path, threads=2):
        from .runtime import validate_threads
        self.threads = validate_threads(threads)
        self.path = Path(path)
        from .bundle_io import model_inventory
        model_inventory(path)
        self.fingerprint = directory_fingerprint(path)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                                         clean_up_tokenization_spaces=True)
        model = AutoModelForQuestionAnswering.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, use_safetensors=True).cpu().eval()
        self.pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=-1)

    def answer(self, question, documents):
        return self.answer_batch(question, documents)

    def answer_batch(self, question, documents, batch_size=8, strategy="max_score"):
        return self.trace(question, documents, batch_size, strategy)["selected"]

    def trace(self, question, documents, batch_size=8, strategy="max_score"):
        from .runtime import cpu_threads
        from .reading import validate_reader_result
        validate_request(question, 1)
        if type(batch_size) is not int or not 1 <= batch_size <= 128:
            raise ValueError("reader batch size must be in [1,128]")
        documents = list(documents)
        if not documents:
            from .reading import select_answer
            return {"selected": select_answer([], [], strategy), "candidates": []}
        candidates = []
        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            with cpu_threads(self.threads), torch.inference_mode():
                outputs = self.pipeline([{"question": question, "context": doc.text} for doc in batch],
                                        top_k=1, max_answer_len=40, max_seq_len=384,
                                        handle_impossible_answer=False, batch_size=batch_size)
            if isinstance(outputs, dict) and len(batch) == 1:
                outputs = [outputs]
            if not isinstance(outputs, list) or len(outputs) != len(batch):
                raise ValueError("reader output batch cardinality mismatch")
            candidates.extend(validate_reader_result(doc, result) for doc, result in zip(batch, outputs))
        from .reading import select_answer
        selected = select_answer(candidates, [d.doc_id for d in documents], strategy)
        return {"selected": selected,
                "candidates": [{**row, "retrieval_rank": rank,
                                "selected": row["document_id"] == selected["document_id"]}
                               for rank, row in enumerate(candidates, 1)]}


class QABundle:
    def __init__(self, documents, encoder, reader, threshold=0.0, reader_k=3, embeddings=None):
        validate_request("configuration", reader_k)
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite")
        self.documents, self.encoder, self.reader = list(documents), encoder, reader
        self.threshold, self.reader_k = float(threshold), reader_k
        self.index = DenseIndex(self.documents, encoder, embeddings)
        self.lookup = {d.doc_id: d for d in self.documents}

    def raw_predict(self, question, retriever=None):
        validate_request(question, self.reader_k)
        started = time.perf_counter()
        from .retrieval import validate_ranked
        engine = self.index if retriever is None else retriever
        ranked = validate_ranked(engine.search(question, max(5, self.reader_k)),
                                 self.lookup, max(5, self.reader_k))
        retrieved = time.perf_counter()
        output = self.reader.answer(question, [self.lookup[d] for d, _ in ranked[:self.reader_k]])
        finished = time.perf_counter()
        return {**output, "ranked_ids": [d for d, _ in ranked], "ranked_scores": [s for _, s in ranked],
                "retrieval_ms": (retrieved - started) * 1000, "reader_ms": (finished - retrieved) * 1000,
                "total_ms": (finished - started) * 1000}

    def predict(self, question):
        raw = self.raw_predict(question)
        from .core import has_answer
        accepted = has_answer(raw["raw_answer"]) and raw["confidence"] >= self.threshold
        return {"answer": raw["raw_answer"] if accepted else "", "confidence": raw["confidence"],
                "abstained": not accepted, "document_id": raw["document_id"] if accepted else None,
                "start": raw["start"] if accepted else None, "end": raw["end"] if accepted else None,
                "retrieved_ids": json.dumps(raw["ranked_ids"])}

    def save(self, path):
        from .bundle_io import atomic_directory, validate_output_location
        validate_output_location(path, [self.encoder.path, self.reader.path])
        with atomic_directory(path) as staging:
            return self._write_bundle(staging)

    def _write_bundle(self, path):
        (path / "corpus.jsonl").write_text("".join(json.dumps(asdict(d)) + "\n" for d in self.documents))
        np.save(path / "embeddings.npy", self.index.embeddings, allow_pickle=False)
        for role, model in [("encoder", self.encoder), ("reader", self.reader)]:
            from .bundle_io import copy_verified_model
            copy_verified_model(model.path, path / role, model.fingerprint)
        from .bundle_io import model_identity
        encoder_identity = model_identity(self.encoder.fingerprint)
        reader_identity = model_identity(self.reader.fingerprint)
        manifest = {"schema_version": 2, "corpus_sha256": corpus_hash(self.documents),
                    "doc_ids": [d.doc_id for d in self.documents], "encoder_fingerprint": self.encoder.fingerprint,
                    "reader_fingerprint": self.reader.fingerprint,
                    "embeddings_sha256": sha256_file(path / "embeddings.npy"),
                    "threshold": self.threshold, "reader_k": self.reader_k,
                    "encoder_max_tokens": self.encoder.config.max_tokens, "reader_max_tokens": 384,
                    "encoder_config": asdict(self.encoder.config),
                    "encoder_id": encoder_identity["id"], "encoder_revision": encoder_identity["revision"],
                    "reader_id": reader_identity["id"], "reader_revision": reader_identity["revision"]}
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest

    @classmethod
    def load(cls, path):
        path = Path(path)
        from .bundle_io import inspect_bundle, encoder_configuration
        from .core import load_corpus
        manifest = inspect_bundle(path)["manifest"]
        documents = load_corpus(path / "corpus.jsonl")
        config = encoder_configuration(manifest)
        encoder, reader = Encoder(path / "encoder", config=config), Reader(path / "reader")
        embeddings = np.load(path / "embeddings.npy", allow_pickle=False)
        return cls(documents, encoder, reader, manifest["threshold"], manifest["reader_k"], embeddings)
