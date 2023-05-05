from src.helpers import get_negative_binary_example
from src.prompting import Prompt

import random


def is_correct_ranking(
    correct_score: float, false_score: float, bigger_is_better: bool
) -> bool:
    """
    :param correct_score:
    :param false_score:
    :param bigger_is_better: if True, the correct score has to be bigger than the false, otherwise vice-versa
    :return:
    """
    if bigger_is_better:
        return correct_score > false_score
    else:
        return correct_score < false_score


class Task:
    def prepare_for_task(self, example, random_labels: bool):
        """
        Apply the transformations to the example that are necessary for the specific task but which are independent
        of the model / prompt variation used.
        """
        raise NotImplementedError()

    def perform_task(self, model, objective, example, prompt_template, mask_token=None):
        raise NotImplementedError()

    def task_is_correct(self, result, bigger_is_better) -> bool:
        raise NotImplementedError()

    def randomise_labels(self, examples):
        for example in examples:
            assert example["implicature"] in ["yes", "no"], "Cannot randomise labels if label not in 'yes' or 'no'"
            example["implicature"] = random.choice(["yes", "no"])
            example["randomised_label"] = True
            if example["implicature"] != example["type"]:
                example["label_is_random"] = True
            else:
                example["label_is_random"] = False
        return examples


class RankingTask(Task):
    """
    This task prepares a negative and positive example and expects the model to rank them accordingly.
    """

    @staticmethod
    def prepare_task_input(
        correct_example,
        false_example,
        prompt_template: Prompt,
        prompt_examples=None,
        mask_token=None,
    ):
        correct_prompt = prompt_template.prompt(
            correct_example, False, prompt_examples, mask_token=None
        )
        if mask_token:
            masked_input = prompt_template.prompt(
                correct_example, False, prompt_examples, mask_token
            )
        else:
            masked_input = None
        false_prompt = prompt_template.prompt(
            false_example, True, prompt_examples, mask_token=None
        )
        return {
            "correct": correct_prompt,
            "false": false_prompt,
            "masked_input": masked_input,
        }

    @staticmethod
    def prepare_datapoint(
        example,
        prompt_template: Prompt,
        is_false_example: bool,
        prompt_examples=None,
    ):
        label = prompt_template.prompt(
            example, is_false_example, prompt_examples, mask_token=None
        )
        return label

    def prepare_for_task(self, example, random_labels: bool):
        # Unpack examples and get a negative example
        false_example = get_negative_binary_example(example["example"])
        correct_example = example["example"]
        prompt_examples = example["prompts"]
        if random_labels:
            prompt_examples = self.randomise_labels(prompt_examples)
        return {
            "correct_example": correct_example,
            "false_example": false_example,
            "prompt_examples": prompt_examples,
        }

    def perform_task(
        self, model, objective, example, prompt_template: Prompt, mask_token=None
    ):
        prompt_examples = example["prompt_examples"]
        texts_to_score = self.prepare_task_input(
            example["correct_example"],
            example["false_example"],
            prompt_template,
            prompt_examples,
            mask_token,
        )
        correct_model_score = model.get_model_score(
            [
                texts_to_score["masked_input"]
                if objective == "mlm"
                else texts_to_score["correct"]
            ],
            [texts_to_score["correct"]],
            objective,
        )
        false_model_score = model.get_model_score(
            [
                texts_to_score["masked_input"]
                if objective == "mlm"
                else texts_to_score["false"]
            ],
            [texts_to_score["false"]],
            objective,
        )

        return correct_model_score, false_model_score, texts_to_score

    def task_is_correct(self, task_result, bigger_is_better):
        correct_model_score, false_model_score, texts_to_score = task_result
        return is_correct_ranking(
            correct_model_score,
            false_model_score,
            bigger_is_better=bigger_is_better,
        )


class CompletionTask(Task):
    """
    This task prepares a prompt and expects the model to complete it.
    """

    def prepare_for_task(self, example, random_labels: bool):
        """
        Apply the transformations to the example that are necessary for the specific task but which are independent
        of the model / prompt variation used.
        """
        if random_labels:
            prompt_examples = self.randomise_labels(example["prompts"])
        else:
            prompt_examples = example["prompts"]
        return {
            "test_example": example["example"],
            "prompt_examples": prompt_examples,
        }

    @staticmethod
    def prepare_datapoint(
            example,
            prompt_template: Prompt,
            is_false_example: bool,
            prompt_examples=None,
    ):
        label = prompt_template.prompt_for_completion(
            example, prompt_examples, mask_token=None
        )
        return label

    def perform_task(self, model, objective, example, prompt_template, mask_token=None):
        raise NotImplementedError()

    def task_is_correct(self, result, bigger_is_better) -> bool:
        raise NotImplementedError()
