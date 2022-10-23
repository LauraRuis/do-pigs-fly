import pytest

from src.prompting import ExamplePrompt
from src.task import (
    RankingTask,
    is_correct_ranking,
)


@pytest.mark.parametrize(
    "test_input, expected_output",
    [
        (
            {
                "correct_score": 0.0,
                "false_score": 0.0,
                "bigger_is_better": True,
            },
            False,
        ),
        (
            {
                "correct_score": -2.0,
                "false_score": 0.0,
                "bigger_is_better": True,
            },
            False,
        ),
        (
            {
                "correct_score": -2.0,
                "false_score": 0.0,
                "bigger_is_better": False,
            },
            True,
        ),
        (
            {
                "correct_score": 5.0,
                "false_score": 8.0,
                "bigger_is_better": True,
            },
            False,
        ),
        (
            {
                "correct_score": 3.0,
                "false_score": 4.0,
                "bigger_is_better": True,
            },
            False,
        ),
        (
            {
                "correct_score": 3.0,
                "false_score": 5.0,
                "bigger_is_better": False,
            },
            True,
        ),
        (
            {
                "correct_score": 35.0,
                "false_score": 70.0,
                "bigger_is_better": True,
            },
            False,
        ),
        (
            {
                "correct_score": 35.0,
                "false_score": 4.0,
                "bigger_is_better": False,
            },
            False,
        ),
    ],
)
def test_is_correct_ranking(test_input, expected_output):
    assert is_correct_ranking(**test_input) == expected_output


class TestRankingTask:
    ranking_task = RankingTask()

    def test_prepare_task_input(self):
        # Test prepare correct texts if prompt has no instruction texts.
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
        test_prompt = {
            "example_template_text": "Esther asked Juan '%s'. Juan responded '%s'. The answer Juan implied by saying this was %s.",
            "phrase_to_split_on_for_bias_prediction": "responded ",
            "prompt_instruction_text": "The following are examples of the task:",
            "example_instruction_text": "Finish the following text:",
            "phrase_to_put_mask": "was %s",
        }
        test_prompt = ExamplePrompt(**test_prompt, contrastive=False)
        expected_correct_output = (
            "Finish the following text:\n\nEsther asked Juan 'Will these go away? These gas gauges?'. "
            "Juan responded 'It's probably a functional impairment.'. "
            "The answer Juan implied by saying this was no."
        )
        expected_false_output = (
            "Finish the following text:\n\nEsther asked Juan 'Will these go away? These gas gauges?'. "
            "Juan responded 'It's probably a functional impairment.'. "
            "The answer Juan implied by saying this was yes."
        )
        expected_output = {
            "correct": expected_correct_output,
            "false": expected_false_output,
            "masked_input": None,
        }
        actual_output = self.ranking_task.prepare_task_input(
            correct_example, false_example, test_prompt
        )
        assert actual_output == expected_output

        # Test prepare correct texts if prompt has no instruction texts with masking.
        correct_example = {
            "source": "",
            "type": "unknown",
            "utterance": "Are you coming to the party on Friday?",
            "response": "Does the pope shit in the woods?",
            "implicature": "yes",
        }
        false_example = {
            "source": "",
            "type": "unknown",
            "utterance": "Are you coming to the party on Friday?",
            "response": "Does the pope shit in the woods?",
            "implicature": "no",
        }
        expected_correct_output = (
            "Finish the following text:\n\nEsther asked Juan 'Are you coming to the party on Friday?'. "
            "Juan responded 'Does the pope shit in the woods?'. "
            "The answer Juan implied by saying this was yes."
        )
        expected_false_output = (
            "Finish the following text:\n\nEsther asked Juan 'Are you coming to the party on Friday?'. "
            "Juan responded 'Does the pope shit in the woods?'. "
            "The answer Juan implied by saying this was no."
        )
        expected_masked_input = (
            "Finish the following text:\n\nEsther asked Juan 'Are you coming to the party on Friday?'. "
            "Juan responded 'Does the pope shit in the woods?'. "
            "The answer Juan implied by saying this was [mask]."
        )
        expected_output = {
            "correct": expected_correct_output,
            "false": expected_false_output,
            "masked_input": expected_masked_input,
        }
        actual_output = self.ranking_task.prepare_task_input(
            correct_example, false_example, test_prompt, mask_token="[mask]"
        )
        assert actual_output == expected_output

        # Test prepare correct texts if prompt has instruction texts, masking, and prompt examples.
        test_prompt = {
            "prompt_instruction_text": "Each following response to the question implies yes or no:",
            "example_instruction_text": "Does the following response to the question imply yes or no?",
            "example_template_text": "question: %s\nresponse: %s\nimplicature: %s",
            "phrase_to_split_on_for_bias_prediction": "response: ",
            "phrase_to_put_mask": "implicature: %s",
        }
        expected_correct_output = (
            "Each following response to the question implies yes or no:\n\n"
            "question: Are you going to the party on Friday?\n"
            "response: Is the pope catholic?\n"
            "implicature: Yes.\n\n"
            "question: this is a test. \n"
            "response: do chicken have lips\n"
            "implicature: no\n\n"
            "Does the following response to the question imply yes or no?\n\n"
            "question: this is a test. \n"
            "response: do chicken have lips\n"
            "implicature: no"
        )
        expected_false_output = (
            "Each following response to the question implies yes or no:\n\n"
            "question: Are you going to the party on Friday?\n"
            "response: Is the pope catholic?\n"
            "implicature: Yes.\n\n"
            "question: this is a test. \n"
            "response: do chicken have lips\n"
            "implicature: no\n\n"
            "Does the following response to the question imply yes or no?\n\n"
            "question: this is a test. \n"
            "response: do chicken have lips\n"
            "implicature: yes"
        )
        expected_masked_input = (
            "Each following response to the question implies yes or no:\n\n"
            "question: Are you going to the party on Friday?\n"
            "response: Is the pope catholic?\n"
            "implicature: Yes.\n\n"
            "question: this is a test. \n"
            "response: do chicken have lips\n"
            "implicature: no\n\n"
            "Does the following response to the question imply yes or no?\n\n"
            "question: this is a test. \n"
            "response: do chicken have lips\n"
            "implicature: {{mask}}"
        )
        expected_output = {
            "correct": expected_correct_output,
            "false": expected_false_output,
            "masked_input": expected_masked_input,
        }
        prompt_example = {
            "utterance": "Are you going to the party on Friday?",
            "response": "Is the pope catholic?",
            "implicature": "Yes.",
        }
        correct_example = {
            "utterance": "this is a test. ",
            "response": "do chicken have lips",
            "implicature": "no",
        }
        false_example = {
            "utterance": "this is a test. ",
            "response": "do chicken have lips",
            "implicature": "yes",
        }
        test_prompt_template = ExamplePrompt(**test_prompt, contrastive=False)
        actual_output = self.ranking_task.prepare_task_input(
            correct_example,
            false_example,
            test_prompt_template,
            prompt_examples=[prompt_example, correct_example],
            mask_token="{{mask}}",
        )
        assert expected_output == actual_output
