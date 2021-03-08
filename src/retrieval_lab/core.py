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
import unicodedata


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str

    def __post_init__(self):
        validate_identifier(self.doc_id)
        _text(self.title, "title")
        _text(self.text, "text")


@dataclass(frozen=True)
class Query:
    query_id: str
    question: str
    relevant_ids: tuple[str, ...]
    answers: tuple[str, ...]

    def __post_init__(self):
        validate_identifier(self.query_id)
        _text(self.question, "question", 2048)
        for field in [self.relevant_ids, self.answers]:
            if not isinstance(field, tuple) or any(not isinstance(x, str) or not x.strip() for x in field):
                raise ValueError("query labels must be tuples of nonempty strings")
            if len(set(field)) != len(field):
                raise ValueError("duplicate query labels")
        if bool(self.relevant_ids) != bool(self.answers):
            raise ValueError("answerable queries need answers and relevant IDs")
        for identifier in self.relevant_ids:
            validate_identifier(identifier)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def corpus_hash(documents):
    return canonical_hash([asdict(d) for d in documents])


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonstandard JSON number: {value}")


def _jsonl(path, max_bytes=16 * 1024 * 1024, max_records=100000):
    if type(max_bytes) is not int or max_bytes < 1 or type(max_records) is not int or max_records < 1:
        raise ValueError("positive byte and record limits required")
    values, byte_count = [], 0
    with Path(path).open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            byte_count += len(line)
            if byte_count > max_bytes:
                raise ValueError("dataset byte budget exceeded")
            if not line.strip():
                continue
            if len(values) >= max_records:
                raise ValueError("dataset record budget exceeded")
            try:
                item = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_object,
                                  parse_constant=_reject_constant)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSONL: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: JSONL records must be objects")
            values.append(item)
    if not values:
        raise ValueError("JSONL must contain at least one object")
    return values


def _text(value, field, limit=10000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{field} must be a nonempty string of at most {limit} characters")
    return value


def validate_identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value):
        raise ValueError("identifier must be 1-128 ASCII letters, digits, dots, colons, underscores or hyphens")
    return value


def load_corpus(path):
    documents, seen = [], set()
    for item in _jsonl(path):
        if set(item) != {"doc_id", "title", "text"}:
            raise ValueError("document fields must be doc_id, title, text")
        document = Document(**{key: _text(value, key) for key, value in item.items()})
        validate_identifier(document.doc_id)
        if document.doc_id in seen:
            raise ValueError(f"duplicate document ID: {document.doc_id}")
        seen.add(document.doc_id)
        documents.append(document)
    return documents


def question_key(question):
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def load_queries(path, documents):
    queries, seen, seen_questions = [], set(), set()
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
        validate_identifier(item["query_id"])
        if item["query_id"] in seen:
            raise ValueError("duplicate query ID")
        seen.add(item["query_id"])
        key = question_key(item["question"])
        if key in seen_questions:
            raise ValueError("duplicate question within split")
        seen_questions.add(key)
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
    questions = {question_key(q.question) for q in dev} & {question_key(q.question) for q in test}
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
    def __init__(self, documents, k1=1.5, b=0.75, tokenizer=None, title_weight=1.0):
        self.documents = list(documents)
        if len({d.doc_id for d in self.documents}) != len(self.documents):
            raise ValueError("duplicate document ID")
        if (type(k1) not in (int, float) or not math.isfinite(k1) or k1 <= 0
                or type(b) not in (int, float) or not math.isfinite(b) or not 0 <= b <= 1):
            raise ValueError("BM25 requires finite k1 > 0 and b in [0,1]")
        self.k1, self.b = k1, b
        self.tokenizer = tokenizer if tokenizer is not None else tokens
        if not callable(self.tokenizer):
            raise ValueError("tokenizer must be callable")
        if type(title_weight) not in (int, float) or not math.isfinite(title_weight) or title_weight < 0:
            raise ValueError("title weight must be finite and nonnegative")
        self.title_weight = title_weight
        self.counts = []
        for document in self.documents:
            counts = Counter(self.tokenizer(document.text))
            for token, frequency in Counter(self.tokenizer(document.title)).items():
                if title_weight:
                    counts[token] += frequency * title_weight
            self.counts.append(counts)
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg_length = statistics.mean(self.lengths) if self.lengths else 0
        self.df = Counter(word for counts in self.counts for word in counts)
        self.postings = {}
        for index, counts in enumerate(self.counts):
            for word in counts:
                self.postings.setdefault(word, set()).add(index)

    def candidate_ids(self, question):
        validate_request(question, 1)
        indices = set().union(*(self.postings.get(word, set()) for word in self.tokenizer(question)))
        return [self.documents[index].doc_id for index in sorted(indices)]

    def explain(self, question, doc_id):
        validate_request(question, 1)
        positions = {d.doc_id: i for i, d in enumerate(self.documents)}
        if doc_id not in positions:
            raise ValueError("unknown document ID")
        index = positions[doc_id]
        counts, length, terms = self.counts[index], self.lengths[index], []
        for word in sorted(set(self.tokenizer(question))):
            tf, contribution = counts[word], 0.0
            idf = math.log(1 + (len(self.documents) - self.df[word] + 0.5) / (self.df[word] + 0.5))
            if tf:
                contribution = idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * length / self.avg_length))
            terms.append({"term": word, "frequency": tf, "idf": idf, "contribution": contribution})
        return {"doc_id": doc_id, "terms": terms, "score": sum(x["contribution"] for x in terms)}

    def search(self, question, k=5):
        validate_request(question, k)
        n = len(self.documents)
        if not n:
            return []
        ranked = []
        candidates = set(self.candidate_ids(question))
        for document, counts, length in zip(self.documents, self.counts, self.lengths):
            if document.doc_id not in candidates:
                continue
            score = 0.0
            for word in set(self.tokenizer(question)):
                tf = counts[word]
                if tf:
                    idf = math.log(1 + (n - self.df[word] + 0.5) / (self.df[word] + 0.5))
                    score += idf * tf * (self.k1 + 1) / (
                        tf + self.k1 * (1 - self.b + self.b * length / self.avg_length))
            if score > 0:
                ranked.append((document.doc_id, score))
        from .retrieval import stable_top_k
        return stable_top_k(ranked, k)


def validate_index(manifest, documents, encoder_fingerprint):
    expected = {"corpus_sha256": corpus_hash(documents), "doc_ids": [d.doc_id for d in documents],
                "encoder_fingerprint": encoder_fingerprint}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"index mismatch: {key}")


def normalize(text, policy="squad"):
    if not isinstance(text, str) or policy not in {"squad", "unicode"}:
        raise ValueError("normalization requires text and a known policy")
    if policy == "unicode":
        text = unicodedata.normalize("NFKC", text).casefold()
        text = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_scores(answer, references, policy="squad"):
    reference_tokens = [normalize(g, policy).split() for g in references] or [[]]
    predicted = normalize(answer, policy).split()
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
    seen_query_ids = set()
    required = {"query_id", "raw_answer", "confidence", "answers", "ranked_ids", "relevant_ids",
                "retrieval_ms", "reader_ms", "total_ms"}
    for row in rows:
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError("evaluation row is missing required fields")
        validate_identifier(row["query_id"])
        if row["query_id"] in seen_query_ids:
            raise ValueError("duplicate evaluation query ID")
        seen_query_ids.add(row["query_id"])
        if not isinstance(row["raw_answer"], str):
            raise ValueError("raw answer must be a string")
        score = row["confidence"]
        if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("confidence must be finite and in [0,1]")
        for field in ["answers", "ranked_ids", "relevant_ids"]:
            values = row[field]
            if not isinstance(values, (list, tuple)) or any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError(f"{field} must contain nonempty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate {field}")
        if bool(row["answers"]) != bool(row["relevant_ids"]):
            raise ValueError("answers and relevance labels disagree")
        for phase in ["retrieval_ms", "reader_ms", "total_ms"]:
            value = row[phase]
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError("latencies must be finite nonnegative numbers")


def has_answer(value):
    return isinstance(value, str) and bool(value.strip())


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
        tpr = statistics.mean(has_answer(r["raw_answer"]) and r["confidence"] >= threshold for r in positives)
        tnr = statistics.mean(not has_answer(r["raw_answer"]) or r["confidence"] < threshold for r in negatives)
        curve.append({"threshold": threshold, "balanced_accuracy": (tpr + tnr) / 2})
    winner = max(curve, key=lambda row: (row["balanced_accuracy"], row["threshold"]))
    return {"split": "dev", "threshold": winner["threshold"],
            "objective": "balanced_answerability_accuracy", "objective_value": winner["balanced_accuracy"],
            "tie_break": "higher_threshold", "query_ids": [r["query_id"] for r in rows], "curve": curve}


def percentile(values, fraction):
    values = list(values)
    if (not values or type(fraction) not in (int, float) or not 0 <= fraction <= 1
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
        raise ValueError("percentile requires finite samples and a fraction in [0,1]")
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def score_predictions(rows, threshold, reader_k=3, *, expected_query_ids=None, cutoffs=(1, 3, 5), normalization="squad"):
    _validate_rows(rows)
    if expected_query_ids is not None:
        expected = list(expected_query_ids)
        if len(set(expected)) != len(expected) or set(expected) != {r["query_id"] for r in rows}:
            raise ValueError("evaluation query coverage mismatch")
    validate_request("reader candidates", reader_k)
    cutoffs = tuple(cutoffs)
    if not cutoffs or len(set(cutoffs)) != len(cutoffs):
        raise ValueError("metric cutoffs must be nonempty and unique")
    for cutoff in cutoffs:
        validate_request("metric cutoff", cutoff)
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1.0000001:
        raise ValueError("threshold must be finite and within the score range or abstain-all sentinel")
    normalize("", normalization)
    scored = []
    for row in rows:
        answer = row["raw_answer"] if has_answer(row["raw_answer"]) and row["confidence"] >= threshold else ""
        em, f1 = answer_scores(answer, row["answers"], normalization)
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
    for k in sorted(cutoffs):
        recalls = [len(set(r["ranked_ids"][:k]) & set(r["relevant_ids"])) / len(r["relevant_ids"])
                   for r in answerable]
        metrics[f"recall_at_{k}"] = statistics.mean(recalls) if recalls else None
    reciprocal_ranks = [next((1 / (i + 1) for i, d in enumerate(r["ranked_ids"][:max(cutoffs)])
                              if d in r["relevant_ids"]), 0.0) for r in answerable]
    metrics[f"mrr_at_{max(cutoffs)}"] = statistics.mean(reciprocal_ranks) if reciprocal_ranks else None
    accepted = [r for r in scored if r["answer"]]
    metrics["selective_em"] = statistics.mean(r["answer_em"] for r in accepted) if accepted else None
    metrics["retrieval_query_count"] = len(answerable)
    metrics["accepted_answer_count"] = len(accepted)
    for phase in ["retrieval", "reader", "total"]:
        values = [r[f"{phase}_ms"] for r in rows]
        metrics[f"latency_{phase}_p50_ms"] = percentile(values, 0.5)
        metrics[f"latency_{phase}_p95_ms"] = percentile(values, 0.95)
    for category in ["correct", "reader_failure", "retrieval_miss", "false_abstention",
                     "correct_abstention", "unanswerable_false_positive"]:
        metrics[f"count_{category}"] = sum(r["failure"] == category for r in scored)
    return metrics, scored


def validate_evidence_isolation(dev, test, documents):
    lookup = {d.doc_id: question_key(d.text) for d in documents}
    left = {lookup[d] for q in dev for d in q.relevant_ids}
    right = {lookup[d] for q in test for d in q.relevant_ids}
    if left & right:
        raise ValueError("evidence content overlap across splits")
