"""Validated data, BM25, and metrics; no network or model dependencies."""
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import string


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Query:
    query_id: str
    question: str
    relevant_ids: tuple[str, ...]
    answers: tuple[str, ...]


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def corpus_hash(documents):
    return canonical_hash([asdict(d) for d in documents])


def _jsonl(path):
    try:
        values = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"malformed JSONL in {path}") from exc
    if not values or any(not isinstance(item, dict) for item in values):
        raise ValueError("JSONL must contain at least one object")
    return values


def _text(value, field, limit=10000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{field} must be a nonempty string of at most {limit} characters")
    return value


def load_corpus(path):
    documents, seen = [], set()
    for item in _jsonl(path):
        if set(item) != {"doc_id", "title", "text"}:
            raise ValueError("document fields must be doc_id, title, text")
        document = Document(**{key: _text(value, key) for key, value in item.items()})
        if document.doc_id in seen:
            raise ValueError(f"duplicate document ID: {document.doc_id}")
        seen.add(document.doc_id)
        documents.append(document)
    return documents


def load_queries(path, documents):
    queries, seen = [], set()
    lookup = {d.doc_id: d for d in documents}
    for item in _jsonl(path):
        if set(item) != {"query_id", "question", "relevant_ids", "answers"}:
            raise ValueError("query fields must be query_id, question, relevant_ids, answers")
        for key in ["query_id", "question"]:
            _text(item[key], key, 2048)
        for key in ["relevant_ids", "answers"]:
            if not isinstance(item[key], list):
                raise ValueError(f"{key} must be a list")
            for value in item[key]:
                _text(value, key)
            if len(set(item[key])) != len(item[key]):
                raise ValueError(f"duplicate {key}")
        if item["query_id"] in seen:
            raise ValueError("duplicate query ID")
        seen.add(item["query_id"])
        if set(item["relevant_ids"]) - lookup.keys():
            raise ValueError("query references unknown document")
        if bool(item["answers"]) != bool(item["relevant_ids"]):
            raise ValueError("answerable questions need both answers and relevant document IDs")
        for answer in item["answers"]:
            if not any(answer.lower() in lookup[d].text.lower() for d in item["relevant_ids"]):
                raise ValueError("gold answer is absent from relevant document text")
        queries.append(Query(item["query_id"], item["question"], tuple(item["relevant_ids"]),
                             tuple(item["answers"])))
    return queries


def validate_splits(dev, test):
    ids = {q.query_id for q in dev} & {q.query_id for q in test}
    questions = {normalize(q.question) for q in dev} & {normalize(q.question) for q in test}
    if ids or questions:
        raise ValueError("development/test overlap detected")


def validate_request(question, k):
    _text(question, "question", 2048)
    if type(k) is not int or not 1 <= k <= 100:
        raise ValueError("k must be an integer between 1 and 100")


def tokens(text):
    return re.findall(r"\b\w+\b", text.lower())


class BM25:
    """Okapi BM25 with positive Robertson IDF; deterministic document-ID ties."""
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = list(documents)
        if len({d.doc_id for d in documents}) != len(documents):
            raise ValueError("duplicate document ID")
        self.k1, self.b = k1, b
        self.counts = [Counter(tokens(d.title + " " + d.text)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg_length = statistics.mean(self.lengths) if self.lengths else 0
        self.df = Counter(word for counts in self.counts for word in counts)

    def search(self, question, k=5):
        validate_request(question, k)
        n = len(self.documents)
        if not n:
            return []
        ranked = []
        for document, counts, length in zip(self.documents, self.counts, self.lengths):
            score = 0.0
            for word in set(tokens(question)):
                tf = counts[word]
                if tf:
                    idf = math.log(1 + (n - self.df[word] + 0.5) / (self.df[word] + 0.5))
                    score += idf * tf * (self.k1 + 1) / (
                        tf + self.k1 * (1 - self.b + self.b * length / self.avg_length))
            if score > 0:
                ranked.append((document.doc_id, score))
        return sorted(ranked, key=lambda pair: (-pair[1], pair[0]))[:k]


def validate_index(manifest, documents, encoder_fingerprint):
    expected = {"corpus_sha256": corpus_hash(documents), "doc_ids": [d.doc_id for d in documents],
                "encoder_fingerprint": encoder_fingerprint}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"index mismatch: {key}")


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_scores(answer, references):
    reference_tokens = [normalize(g).split() for g in references] or [[]]
    predicted = normalize(answer).split()
    exact, f1 = 0.0, 0.0
    for gold in reference_tokens:
        exact = max(exact, float(predicted == gold))
        overlap = sum((Counter(predicted) & Counter(gold)).values())
        current = float(not predicted and not gold)
        if overlap:
            current = 2 * overlap / (len(predicted) + len(gold))
        f1 = max(f1, current)
    return exact, f1


def _validate_rows(rows):
    if not rows:
        raise ValueError("evaluation requires nonempty rows")
    for row in rows:
        score = row["confidence"]
        if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("confidence must be finite and in [0,1]")
        if len(set(row["ranked_ids"])) != len(row["ranked_ids"]):
            raise ValueError("duplicate retrieved document IDs")


def calibrate_threshold(rows, *, split):
    """Maximize dev balanced answerability accuracy; higher threshold wins ties."""
    if split != "dev":
        raise ValueError("threshold selection requires split='dev'")
    _validate_rows(rows)
    positives = [r for r in rows if r["answers"]]
    negatives = [r for r in rows if not r["answers"]]
    if not positives or not negatives:
        raise ValueError("development calibration needs answerable and unanswerable questions")
    candidates = sorted({0.0, 1.0000001, *(r["confidence"] for r in rows)})
    curve = []
    for threshold in candidates:
        tpr = statistics.mean(bool(r["raw_answer"]) and r["confidence"] >= threshold for r in positives)
        tnr = statistics.mean(not r["raw_answer"] or r["confidence"] < threshold for r in negatives)
        curve.append({"threshold": threshold, "balanced_accuracy": (tpr + tnr) / 2})
    winner = max(curve, key=lambda row: (row["balanced_accuracy"], row["threshold"]))
    return {"split": "dev", "threshold": winner["threshold"],
            "objective": "balanced_answerability_accuracy", "objective_value": winner["balanced_accuracy"],
            "tie_break": "higher_threshold", "query_ids": [r["query_id"] for r in rows], "curve": curve}


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def score_predictions(rows, threshold, reader_k=3):
    _validate_rows(rows)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    scored = []
    for row in rows:
        answer = row["raw_answer"] if row["confidence"] >= threshold else ""
        em, f1 = answer_scores(answer, row["answers"])
        relevant = set(row["relevant_ids"])
        if not relevant:
            failure = "correct_abstention" if not answer else "unanswerable_false_positive"
        elif not relevant.intersection(row["ranked_ids"][:reader_k]):
            failure = "retrieval_miss"
        elif not answer:
            failure = "false_abstention"
        else:
            failure = "correct" if em else "reader_failure"
        scored.append({**row, "answer": answer, "abstained": not bool(answer),
                       "answer_em": em, "answer_f1": f1, "failure": failure})
    answerable = [r for r in rows if r["relevant_ids"]]
    metrics = {"query_count": len(rows), "answerable_count": len(answerable),
               "answer_em": statistics.mean(r["answer_em"] for r in scored),
               "answer_f1": statistics.mean(r["answer_f1"] for r in scored),
               "coverage": statistics.mean(bool(r["answer"]) for r in scored)}
    for k in [1, 3, 5]:
        recalls = [len(set(r["ranked_ids"][:k]) & set(r["relevant_ids"])) / len(r["relevant_ids"])
                   for r in answerable]
        metrics[f"recall_at_{k}"] = statistics.mean(recalls) if recalls else 0.0
    reciprocal_ranks = [next((1 / (i + 1) for i, d in enumerate(r["ranked_ids"][:5])
                              if d in r["relevant_ids"]), 0.0) for r in answerable]
    metrics["mrr_at_5"] = statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    accepted = [r for r in scored if r["answer"]]
    metrics["selective_em"] = statistics.mean(r["answer_em"] for r in accepted) if accepted else 0.0
    for phase in ["retrieval", "reader", "total"]:
        values = [r[f"{phase}_ms"] for r in rows]
        metrics[f"latency_{phase}_p50_ms"] = percentile(values, 0.5)
        metrics[f"latency_{phase}_p95_ms"] = percentile(values, 0.95)
    for category in ["correct", "reader_failure", "retrieval_miss", "false_abstention",
                     "correct_abstention", "unanswerable_false_positive"]:
        metrics[f"count_{category}"] = sum(r["failure"] == category for r in scored)
    return metrics, scored
