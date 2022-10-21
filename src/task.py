from src.helpers import get_negative_binary_example
from src.prompting import Prompt


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
    def prepare_for_task(self, example):
        """
        Apply the transformations to the example that are necessary for the specific task but which are independent
        of the model / prompt variation used.
        """
        raise NotImplementedError()

    def perform_task(self, model, objective, example, prompt_template, mask_token=None):
        raise NotImplementedError()

    def task_is_correct(self, result, bigger_is_better) -> bool:
        raise NotImplementedError()


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
            correct_example, prompt_examples, mask_token=None
        )
        if mask_token:
            masked_input = prompt_template.prompt(
                correct_example, prompt_examples, mask_token
            )
        else:
            masked_input = None
        false_prompt = prompt_template.prompt(
            false_example, prompt_examples, mask_token=None
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
        label = prompt_template.prompt(example, is_false_example, prompt_examples, mask_token=None)
        return label

    def prepare_for_task(self, example):
        # Unpack examples and get a negative example
        false_example = get_negative_binary_example(example["example"])
        correct_example = example["example"]
        prompt_examples = example["prompts"]
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
            [texts_to_score["masked_input"] if objective == "mlm" else texts_to_score["correct"]],
            [texts_to_score["correct"]],
            objective,
        )
        false_model_score = model.get_model_score(
            [texts_to_score["masked_input"] if objective == "mlm" else texts_to_score["false"]],
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
