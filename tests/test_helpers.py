import pytest
from src.helpers import (
    AccuracyMetric,
    MultiMetric,
    get_negative_binary_example,
)

import numpy as np


class TestRankingMetric:
    metric = AccuracyMetric(name="test metric")

    def test_reset(self):
        correct = 1
        self.metric.update(correct)
        assert self.metric._correct == 1.0
        assert self.metric._total == 1.0
        self.metric.reset()
        assert self.metric._correct == 0
        assert self.metric._total == 0

    def test_get_accuracy(self):
        correct = 1.0
        self.metric.update(correct)
        assert self.metric.get_accuracy == 100.0
        correct = 0.0
        self.metric.update(correct)
        assert self.metric.get_accuracy == 50.0
        correct = 0
        self.metric.update(correct)
        assert self.metric.get_accuracy == 33.33333333333333
        self.metric.reset()
        with pytest.raises(Exception):
            self.metric.get_accuracy

    def test_update(self):
        correct = 1
        self.metric.update(correct)
        assert self.metric._correct == 1
        assert self.metric._total == 1
        correct = 0.0
        self.metric.update(correct)
        assert self.metric._correct == 1
        assert self.metric._total == 2

class TestMultiMetric:
    metrics = MultiMetric(num_metrics=3)

    def test_reset(self):
        correct = 1
        self.metrics.update(0, correct)
        self.metrics.update(1, correct)
        self.metrics.update(2, correct)
        assert self.metrics._metrics[0]._correct == 1.0
        assert self.metrics._metrics[0]._total == 1.0
        assert self.metrics._metrics[1]._correct == 1.0
        assert self.metrics._metrics[1]._total == 1.0
        assert self.metrics._metrics[2]._correct == 1.0
        assert self.metrics._metrics[2]._total == 1.0
        self.metrics.reset()
        assert self.metrics._metrics[0]._correct == 0
        assert self.metrics._metrics[0]._total == 0
        assert self.metrics._metrics[1]._correct == 0
        assert self.metrics._metrics[1]._total == 0

    def test_update(self):
        correct = 1
        actual_correct = self.metrics.update(0, correct)
        expected_accuracy = 100.0
        expected_correct = True
        assert expected_accuracy == self.metrics.get_accuracy(0)
        assert expected_correct == actual_correct
        expected_accuracy = 100.0
        expected_correct = True
        actual_correct = self.metrics.update(1, correct)
        assert expected_accuracy == self.metrics.get_accuracy(1)
        assert expected_correct == actual_correct
        expected_correct = True
        actual_correct = self.metrics.update(2, correct)
        assert expected_accuracy == self.metrics.get_accuracy(2)
        assert expected_correct == actual_correct
        expected_accuracy = 50.0
        correct = 0
        actual_correct = self.metrics.update(0, correct)
        expected_correct = False
        assert expected_accuracy == self.metrics.get_accuracy(0)
        assert expected_correct == actual_correct
        expected_accuracy = 100.0
        correct = 1
        expected_correct = True
        actual_correct = self.metrics.update(1, correct)
        assert expected_accuracy == self.metrics.get_accuracy(1)
        assert expected_correct == actual_correct
        self.metrics.reset()

    def test_mean_and_std(self):
        all_correct = [0, 0, 1]
        for i, correct in enumerate(all_correct):
            self.metrics.update(i, correct)
        expected_accuracies = [0.0, 0.0, 100.0]
        expected_mean = np.mean(expected_accuracies)
        expected_std = np.std(expected_accuracies)
        actual_mean, actual_std = self.metrics.get_mean_and_std()
        assert expected_mean == actual_mean
        assert expected_std == actual_std
        all_correct = [0, 1, 0]
        for i, correct in enumerate(all_correct):
            self.metrics.update(i, correct)
        expected_accuracies = [0.0, 50.0, 50.0]
        expected_mean = np.mean(expected_accuracies)
        expected_std = np.std(expected_accuracies)
        actual_mean, actual_std = self.metrics.get_mean_and_std()
        assert expected_mean == actual_mean
        assert expected_std == actual_std


def test_get_negative_example():
    # Test negation from no to yes
    correct_example = {
        "source": "",
        "type": "unknown",
        "utterance": "Will these go away? These gas gauges?",
        "response": "It's probably a functional impairment.",
        "implicature": "no",
    }
    false_example = {
        "source": "",
        "type": "unknown",
        "utterance": "Will these go away? These gas gauges?",
        "response": "It's probably a functional impairment.",
        "implicature": "yes",
    }
    assert get_negative_binary_example(correct_example) == false_example

    # Test negation from yes to no
    assert get_negative_binary_example(false_example) == correct_example

    # Test fails if unrecognized implicature.
    with pytest.raises(ValueError):
        wrong_example = {
            "source": "",
            "type": "unknown",
            "utterance": "Will these go away? These gas gauges?",
            "response": "It's probably a functional impairment.",
            "implicature": "No.",
        }
        get_negative_binary_example(wrong_example)
