"""Executable integration contract for the separately pinned legacy API."""
import unittest
from pathlib import Path
from runner import evaluate


class HaystackIntegrationTest(unittest.TestCase):
    def test_real_legacy_retriever_reader_experiment(self):
        report = evaluate(Path('.'), Path('evidence/haystack1.json'))
        self.assertEqual(report['status'], 'executed')
        self.assertEqual(report['packages']['farm-haystack'], '1.26.3')
        self.assertEqual(report['packages']['transformers'], '4.39.3')
        self.assertEqual(set(report['experiments']), {'bm25', 'dense'})
        for result in report['experiments'].values():
            self.assertEqual(result['test']['metrics']['query_count'], 12)
            self.assertGreater(result['test']['metrics']['recall_at_3'], 0.5)
            self.assertEqual(result['calibration']['split'], 'dev')


if __name__ == '__main__':
    unittest.main()
