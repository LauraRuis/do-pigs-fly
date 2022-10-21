from collections import defaultdict
from itertools import islice
import numpy as np
import logging
import random
import copy
import json

logger = logging.getLogger()


class MultiMetric:
    """
    Keeps multiple metrics and reports mean/std. dev. between them as well as individual metrics.
    """

    def __init__(self, num_metrics: int):
        """
        :param num_metrics: how many metrics to keep
        :param bigger_is_better: are the metrics such that a bigger score is better or not
        """
        self._metrics = [
            AccuracyMetric(name="Accuracy Metric") for _ in range(num_metrics)
        ]

    def update(self, index_to_update: int, correct: int, **kwargs):
        """

        :param index_to_update: which metric to update
        :param correct: whether the example was correctly predicted according to the task
        :return: whether the example was correctly predicted according to the task
        """
        correct = self._metrics[index_to_update].update(correct=correct)
        return correct

    def get_accuracy(self, index: int):
        return self._metrics[index].get_accuracy

    def get_mean_and_std(self):
        accuracies = [metric.get_accuracy for metric in self._metrics]
        return float(np.mean(accuracies)), float(np.std(accuracies))

    def reset(self):
        for metric in self._metrics:
            metric.reset()

    def __str__(self):
        mean, std = self.get_mean_and_std()
        full_str = "Mean accuracy: %.2f\nStd. Dev: %.2f\nAccuracy per metric:\n" % (
            mean,
            std,
        )
        for metric in self._metrics:
            full_str += str(metric)
            full_str += "\n"
        return full_str


class AccuracyMetric:
    """
    Keeps accuracy metric
    """

    def __init__(self, name: str):
        self._correct = 0
        self._total = 0
        self._name = name

    def reset(self):
        self._correct = 0
        self._total = 0

    @property
    def get_accuracy(self):
        assert self._total, "Can't get accuracy with total=0."
        return (self._correct / self._total) * 100.0

    def update(self, correct):
        assert 0 <= correct <= 1, (
            "correct must be between 0 and 1 but is %.2f" % correct
        )
        self._correct += int(correct)
        self._total += 1
        return correct

    def __str__(self):
        metric_str = "%s accuracy: %.2f\n" % (self._name, self.get_accuracy)
        return metric_str


def get_negative_binary_example(example):
    """
    Creates a false example for a binary implicature example.
    :param example:
    :return: the same dict as the input except for the implicature is negated (yes to no and vice-versa)
    """
    if example["implicature"] == "yes":
        false_implicature = "no"
    elif example["implicature"] == "no":
        false_implicature = "yes"
    else:
        raise ValueError("Unknown implicature %s" % example["implicature"])
    false_example = copy.deepcopy(example)
    false_example["implicature"] = false_implicature
    return false_example


def log_all_results(
    models
):
    for model_d in models:
        logger.info(f"Scores for model card {model_d['model_id']}")
        logger.info("Implicature score:")
        logger.info(model_d["implicature_metrics"])


def log_prompt_templates(prompt_templates, k_shot, mask_token=None):
    prompt_example = {
        "utterance": "Are you going to the party on Friday?",
        "response": "Is the pope catholic?",
        "implicature": "yes",
    }
    prompt_examples = [prompt_example for _ in range(k_shot)]
    test_example = {
        "utterance": "Have you found him yet?",
        "response": "We're still looking.",
        "implicature": "no",
    }
    for i, prompt_template in enumerate(prompt_templates):
        logger.info("Prompt variation %d:" % i)
        if prompt_template.prompt_instruction_set:
            prompt = prompt_template.prompt(
                test_example, is_false_example=False, prompt_examples=prompt_examples, mask_token=mask_token
            )
            logger.info("\n" + prompt)
        else:
            prompt = prompt_template.prompt(test_example, is_false_example=False, mask_token=mask_token)
            logger.info("\n" + prompt)


def save_results_to_file(
        num_prompt_templates,
        models,
        all_prediction_results,
        implicature_data,
        write_data_to,
        write_results_to,
        arguments):
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    metric_labels = ["implicature_metrics"]
    columns = ["model"]
    for i in range(num_prompt_templates):
        for metric_label in metric_labels:
            columns.append(f"{i}_{metric_label}")
    for model_d in models:
        mean, std = model_d["implicature_metrics"].get_mean_and_std()
        all_results[model_d["model_id"]]["mean_accuracy"] = mean
        all_results[model_d["model_id"]]["std"] = std

        for i in range(num_prompt_templates):
            for metric_label in metric_labels:
                accuracy = model_d[metric_label].get_accuracy(i)
                all_results[model_d["model_id"]][f"prompt_template_{i}"][
                    metric_label
                ] = accuracy

    # Write the data used for the evaluation to json files.
    with open(write_data_to + "_implicature.json", "w") as infile:
        json.dump(implicature_data, infile, indent=4)

    serializable_arguments = dict(arguments)
    all_results["arguments"] = serializable_arguments
    all_results["predictions"] = all_prediction_results
    with open(write_results_to, "w") as infile:
        json.dump(all_results, infile, indent=4)


def chunks(items, chunk_size):
    iterator = iter(items)
    while chunk := list(islice(iterator, chunk_size)):
        yield list(map(list, zip(*chunk)))


def log_config(arguments):
    logger.info("Logging used config:")
    logger.info("-" * 50)
    for argument, value in arguments.items():
        logger.info("{}: {}".format(argument, value))
    logger.info("-" * 50)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


