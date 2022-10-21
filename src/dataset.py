from typing import Dict, Union
from copy import deepcopy
import numpy as np
import copy
import json
import csv
import os


class ImplicatureSuperDataset:
    """
    Class that holds an implicature dataset from a file or object, supports iteration,
    calculates statistics about sources and types of implicature in dataset.

    WARNING: the dataset uses a random number generator. Should be reset between generation
    """

    IMPLICATURE_TYPES = ["Q", "I", "M", "unknown"]

    def __init__(
        self,
        test_input_data_path="",
        dev_input_data_path="",
        extra_types=None,
        seed=0,
    ):
        self._test_input_data_path = test_input_data_path
        self._dev_input_data_path = dev_input_data_path
        self._data = None

        # Initialise variables for keeping track of dataset contents.
        self._data_statistics = {
            "num_examples": 0,
            "examples_per_source": {},
        }
        all_types = self.IMPLICATURE_TYPES
        if extra_types is not None:
            all_types += extra_types
        for type_example in all_types:
            self._data_statistics[type_example] = 0
        self._data_statistics = {
            "test": copy.deepcopy(self._data_statistics),
            "dev": copy.deepcopy(self._data_statistics),
        }
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def _reset_rng(self):
        self.rng = np.random.default_rng(self.seed)

    def _parse_example(self, i: int, k_shot=0, split="test_data"):
        """
        A function to parse the i-th example in self.data["data"]
        :param i: which index to parse
        :param k_shot: how many extra examples to parse from different indices than i
        :return: a parsed example
        """
        if k_shot > 0:
            prompt_indices = self.rng.choice(
                range(len(self._data["dev_data"])), k_shot, replace=False
            )
            prompt_examples = [self._data["dev_data"][j] for j in prompt_indices]
        else:
            prompt_examples = []
        example = self._data[split][i]
        return example, prompt_examples

    def _update_statistics(self, example: Dict[str, str], split: str):
        self._data_statistics[split]["num_examples"] += 1
        if example["source"] not in self._data_statistics[split]["examples_per_source"]:
            self._data_statistics[split]["examples_per_source"][example["source"]] = 0
        self._data_statistics[split]["examples_per_source"][example["source"]] += 1
        type_example = example["type"]
        self._data_statistics[split][type_example] += 1

    @staticmethod
    def _example_str(example: Dict[str, str]) -> str:
        return json.dumps(example, indent=4)

    def read_statistics(self):
        assert len(
            self._data["test_data"]
        ), "Can't read statistics because dataset is empty."
        for example in self._data["test_data"]:
            self._update_statistics(example, "test")
        if len(self._data["dev_data"]):
            for example in self._data["dev_data"]:
                self._update_statistics(example, "dev")
        return

    def get_statistics(self):
        return self._data_statistics.copy()

    def add_example(self, example: Dict[str, str], split: str):
        assert len(
            self._data[f"{split}_data"]
        ), "Can't add a single example to an empty dataset."
        assert f"{split}_data" in self._data, f"Unknown split {split}"
        self._data[f"{split}_data"].append(example)
        self._update_statistics(example, split)

    def set_data(self, data: Dict[str, str]) -> None:
        """Method to read data from object."""
        if not type(data) == dict:
            data = json.dumps(data)
        assert "test_data" in data.keys(), "Missing keys test_data in data."
        assert "dev_data" in data.keys(), "Missing keys dev_data in data."
        self._data = deepcopy(data)
        self.read_statistics()
        return

    def print_example(self, example=None):
        if not example:
            assert self._data is not None, "No data found in class."
            example = self._data["dev_data"][0]
        print(self._example_str(example))

    def get_implicature_iterator(self, k_shot=0, **kwargs):
        assert (
            len(self._data["dev_data"]) >= k_shot
        ), "Cannot get %d prompt examples when dev dataset size " "is %d" % (
            k_shot,
            len(self._data["dev_data"]),
        )
        self._reset_rng()
        for i in range(len(self._data["test_data"])):
            example, prompt_examples = self._parse_example(i, k_shot)
            yield {"example": example, "prompts": prompt_examples}

    def get_example(self, idx: int, split: str, k_shot=0):
        assert (
            len(self._data["dev_data"]) >= k_shot
        ), "Cannot get %d prompt examples when dev dataset size " "is %d" % (
            k_shot,
            len(self._data["dev_data"]),
        )
        example, prompt_examples = self._parse_example(idx, k_shot, split=split)
        return {"example": example, "prompts": prompt_examples}

    def __str__(self):
        if self._data is None:
            number_of_test_examples = 0
            number_of_dev_examples = 0
        else:
            number_of_test_examples = len(self._data["test_data"])
            number_of_dev_examples = len(self._data["dev_data"])
        if not self._test_input_data_path:
            read_data_from = "Added data with .set_data()"
        else:
            read_data_from = "Read data from: %s" % self._test_input_data_path
        example_str = self._example_str(self._data["test_data"][0])
        dataset_str = (
            "%s\n"
            "Number of test examples read: %d\n"
            "Number of dev examples read: %d\n"
            "First example in dataset:\n%s"
            % (
                read_data_from,
                number_of_test_examples,
                number_of_dev_examples,
                example_str,
            )
        )
        return dataset_str

    def __len__(self):
        return len(self._data["test_data"])

    @property
    def dev_size(self):
        return len(self._data["dev_data"])


class ParticularisedImplicatureDataset(ImplicatureSuperDataset):
    """
    Expects data in csv format with the following structure:
    Context utterance,Response utterance,Implicature
    "Have you found him yet?",We're still looking.,No.
    """

    @staticmethod
    def _process_text(text):
        return text.strip("\n")

    def _filter_examples(
        self, input_line: Dict[str, str], source=""
    ) -> Union[None, Dict[str, str]]:
        """
        Takes an input_line from the csv file and filters all examples
        where the implicature is not a simple yes or no.
        :param input_line: a line read from a csv file with data
        :param source: the source of the example
        :return:
        """
        if not input_line:
            return None
        if "yes" in input_line["Implicature"].lower()[:5]:
            implicature = "yes"
        elif "no" in input_line["Implicature"].lower()[:4]:
            implicature = "no"
        else:
            return None
        response = self._process_text(input_line["Response utterance"])
        example = {
            "source": source,
            "type": implicature,
            "utterance": self._process_text(input_line["Context utterance"]),
            "response": response,
            "implicature": implicature,
        }
        return example

    @classmethod
    def read_data_csv(
        cls,
        test_input_data_path: str,
        dev_input_data_path: str,
        seed: int,
        source="",
    ):
        assert os.path.exists(
            test_input_data_path
        ), "No test input data file found at: %s\n" "Current working direction: %s" % (
            test_input_data_path,
            os.getcwd(),
        )
        assert os.path.exists(
            dev_input_data_path
        ), "No dev input data file found at: %s\n" "Current working direction: %s" % (
            dev_input_data_path,
            os.getcwd(),
        )
        with open(test_input_data_path, newline="") as csvfile:
            with open(dev_input_data_path, newline="") as dev_csvfile:
                reader = csv.DictReader(csvfile)
                dev_reader = csv.DictReader(dev_csvfile)
                all_data = {
                    "class": "particular conversational implicature",
                    "test_data": [],
                    "dev_data": [],
                }
                dataset = cls(
                    test_input_data_path,
                    dev_input_data_path,
                    seed=seed,
                    extra_types=["yes", "no"],
                )
                for row in reader:
                    example = dataset._filter_examples(row, source)
                    if example is not None:
                        all_data["test_data"].append(example)
                for row in dev_reader:
                    example = dataset._filter_examples(row, source)
                    if example is not None:
                        all_data["dev_data"].append(example)
                dataset.set_data(all_data)
                return dataset
