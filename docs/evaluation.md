# Evaluation protocol

Every split contains eight answerable and four unanswerable questions. All labels are literal extractive answer spans. Validation rejects unknown document IDs, duplicated IDs, malformed records, answers absent from relevant documents, overlapping normalized questions, overlapping question IDs, and overlapping gold documents across development and test.

## Metrics

| Measure | Definition and denominator |
|---|---|
| Recall@k | Fraction of relevant IDs appearing in the top k, averaged over answerable questions only; k = 1, 3, 5. |
| MRR@5 | Reciprocal rank of the first relevant document within top 5, zero if absent; averaged over answerable questions only. |
| Answer EM | Maximum normalized exact match across acceptable gold answers, averaged over every query. |
| Answer F1 | Maximum token-overlap F1 across acceptable answers, averaged over every query. |
| Unanswerable EM/F1 | Empty prediction scores 1; any nonempty prediction scores 0. |
| Coverage | Fraction of all queries receiving a nonempty answer. |
| Selective EM | EM on accepted answers only; zero when no answers are accepted. |
| Latency p50/p95 | Linearly interpolated percentiles of measured retrieval, reader, and combined milliseconds. |

Answer normalization lowercases, removes ASCII punctuation and English articles, and collapses whitespace. F1 uses token multiplicity rather than a set. Recall@k is a document relevance measure; answer scores measure the accepted answer after abstention.

## Calibration

For each retriever, development examples generate candidate thresholds from observed reader scores, zero, and a value above one to permit rejecting every answer. Choose the threshold with the highest mean of answerable acceptance rate and unanswerable rejection rate. On ties, choose the largest threshold. This calibrates a decision rule; it does not make the reader score a calibrated probability. The complete curve, objective, threshold, split name, and development query IDs are saved.

The same reader processes each of the top three retrieved documents. The highest reader score wins across documents. Retriever scores are not mixed with reader scores because their scales differ. The saved Juniper example demonstrates a weakness of this rule: the correct evidence is available, but another document can produce a higher span score.

Thresholds are selected before held-out predictions are collected. Changing the policy after looking at this test set requires a new held-out split for a fresh evaluation claim.

## Failure attribution

The categories are mutually exclusive and evaluated in this order:

1. For unanswerable queries: `correct_abstention` or `unanswerable_false_positive`.
2. For answerable queries without a gold document among reader candidates: `retrieval_miss`.
3. With gold evidence present but no accepted answer: `false_abstention`.
4. With gold evidence present and a wrong accepted answer: `reader_failure`.
5. Otherwise: `correct`.

Reader failures include cross-document arbitration and span extraction mistakes. Attribution identifies the first observable stage failure; it does not establish a causal counterfactual. Raw answers, accepted answers, confidence, document IDs, ranked scores, and gold labels remain in the report for inspection.

## Reproducibility boundaries

The benchmark captures Python/dependency versions, model revisions and per-file hashes, corpus and split hashes, document order, hardware, thread count, and latency protocol. Latency is based on twelve observations per split with no isolation from host background work. There is no throughput or statistical significance claim. The small, synthetic, English-only corpus is designed to exercise evaluation behavior.
