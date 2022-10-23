import pytest

from src.prompting import ExamplePrompt
import random


class TestPrompt:
    """
    Note that these tests are incredibly slow due to the time it takes to load the large language models below.
    """

    test_prompt_one = {
        "prompt_instruction_text": "Each following response to the question implies yes or no:",
        "example_instruction_text": "Does the following response to the question imply yes or no?",
        "example_template_text": "question: %s\nresponse: %s\nimplicature: %s",
        "phrase_to_split_on_for_bias_prediction": "response: ",
        "phrase_to_put_mask": "implicature: %s",
    }
    test_prompt_two = {
        "prompt_instruction_text": "Is the implied meaning of the following responses yes or no:",
        "example_instruction_text": "Is the implied meaning of the following response yes or no:",
        "example_template_text": "question: %s\nresponse: %s\nmeaning: %s",
        "phrase_to_split_on_for_bias_prediction": "response: ",
        "phrase_to_put_mask": "meaning: %s",
    }
    test_prompt_three = {
        "prompt_instruction_text": "The following are examples of the task:",
        "example_instruction_text": "Finish the following sentence according to the task:",
        "example_template_text": 'Esther asked "%s" and Juan responded "%s", which means %s.',
        "phrase_to_split_on_for_bias_prediction": "responded ",
        "phrase_to_put_mask": "means %s",
    }
    test_prompt_four = {
        "example_template_text": 'Esther asked Juan "%s". Juan responded "%s". '
        "The answer Juan implied by saying this was %s.",
        "phrase_to_split_on_for_bias_prediction": "responded ",
        "phrase_to_put_mask": "was %s",
    }
    test_example_one = {
        "utterance": "Are you going to the party on Friday?",
        "response": "Is the pope catholic?",
        "implicature": "yes",
    }
    test_example_two = {
        "utterance": "this is a test. ",
        "response": "do chicken have lips",
        "implicature": "no",
    }
    test_example_three = {
        "utterance": "Have you found him yet?",
        "response": "We're still looking.",
        "implicature": "no",
    }
    test_example_four = {
        "utterance": "I feel horrible. Debbie was furious that I lost her notes. Do you think I should apologize her"
        " again?",
        "response": "If I were you I would cool off for some days before I talk to her again.",
        "implicature": "no",
    }
    test_example_five = {
        "utterance": "I am going to conduct my psychology experiment this Saturday. I would be having a handful. Would"
        " you help by writing the names of the participants.",
        "response": "I have got some work on my own.",
        "implicature": "no",
    }
    test_example_six = {
        "source": "",
        "type": "unknown",
        "utterance": "Will these go away? These gas gauges?",
        "response": "It's probably a functional impairment.",
        "implicature": "no",
    }
    random.seed(1)

    def test_init(self):
        # Test that the correct assertions are carried out.
        failure_example_templates = [
            "This is a test %s",
            "This is a test %s %s",
            "%s %s %s%s",
        ]
        correct_example_template = "This is a %s test %s%s"
        with pytest.raises(AssertionError):
            for example_template_text in failure_example_templates:
                ExamplePrompt(example_template_text, contrastive=False)
                ExamplePrompt(example_template_text, contrastive=True)
        with pytest.raises(AssertionError):
            phrase_to_put_mask = "testt"
            ExamplePrompt(
                correct_example_template,
                phrase_to_put_mask=phrase_to_put_mask,
                contrastive=False,
            )
            ExamplePrompt(
                correct_example_template,
                phrase_to_put_mask=phrase_to_put_mask,
                contrastive=True,
            )

    def test_wrap_in_template(self):
        random.seed(1)
        expected_wrap = (
            "question: Are you going to the party on Friday?\n"
            "response: Is the pope catholic?\n"
            "implicature: yes"
        )
        expected_wrap_contrastive_one = (
            "A: question: Are you going to the party on Friday?\n"
            "response: Is the pope catholic?\n"
            "implicature: yes\n"
            "B: question: Are you going to the party on Friday?\n"
            "response: Is the pope catholic?\n"
            "implicature: no\n"
            "Question: Which of the above texts is coherent, A or B?\nAnswer: A"
        )
        expected_wrap_contrastive_two = (
            "A: question: Are you going to the party on Friday?\n"
            "response: Is the pope catholic?\n"
            "implicature: no\n"
            "B: question: Are you going to the party on Friday?\n"
            "response: Is the pope catholic?\n"
            "implicature: yes\n"
            "Question: Which of the above texts is coherent, A or B?\nAnswer: B"
        )
        test_prompt = ExamplePrompt(**self.test_prompt_one, contrastive=False)
        contrastive_test_prompt = ExamplePrompt(
            **self.test_prompt_one, contrastive=True
        )
        assert test_prompt._wrap_in_template(self.test_example_one, False, False) == expected_wrap
        actual_contrastive_wrap = contrastive_test_prompt._wrap_in_template(
            self.test_example_one, False, False
        )
        assert (
            actual_contrastive_wrap == expected_wrap_contrastive_one
            or actual_contrastive_wrap == expected_wrap_contrastive_two
        )
        expected_wrap = (
            "question: this is a test. \n"
            "response: do chicken have lips\n"
            "meaning: no"
        )
        test_prompt = ExamplePrompt(**self.test_prompt_two, contrastive=False)
        assert test_prompt._wrap_in_template(self.test_example_two, False, False) == expected_wrap

    def test_add_prompt_examples(self):
        random.seed(1)
        expected_prompt = (
            "Each following response to the question implies yes or no:\n\n"
            "question: Are you going to the party on Friday?\n"
            "response: Is the pope catholic?\n"
            "implicature: yes\n\n"
            "question: this is a test. \n"
            "response: do chicken have lips\n"
            "implicature: no\n\n"
        )
        expected_prompt_contrastive = (
            "Each following response to the question implies yes or no:\n\n"
            "A: question: Are you going to the party on Friday?\n"
            "response: Is the pope catholic?\n"
            "implicature: yes\n"
            "B: question: Are you going to the party on Friday?\n"
            "response: Is the pope catholic?\n"
            "implicature: no\n"
            "Question: Which of the above texts is coherent, A or B?\nAnswer: A\n\n"
            "A: question: this is a test. \n"
            "response: do chicken have lips\n"
            "implicature: no\n"
            "B: question: this is a test. \n"
            "response: do chicken have lips\n"
            "implicature: yes\n"
            "Question: Which of the above texts is coherent, A or B?\nAnswer: A\n\n"
        )
        test_prompt = ExamplePrompt(**self.test_prompt_one, contrastive=False)
        contrastive_test_prompt = ExamplePrompt(
            **self.test_prompt_one, contrastive=True
        )
        test_prompt_examples = [
            test_prompt._wrap_in_template(self.test_example_one, False, True),
            test_prompt._wrap_in_template(self.test_example_two, False, True),
        ]
        assert expected_prompt == test_prompt._add_prompt_examples(test_prompt_examples)
        contrastive_test_prompt_examples = [
            contrastive_test_prompt._wrap_in_template(self.test_example_one, False, True),
            contrastive_test_prompt._wrap_in_template(self.test_example_two, False, True),
        ]
        actual_prompt_contrastive = contrastive_test_prompt._add_prompt_examples(
            contrastive_test_prompt_examples
        )
        assert expected_prompt_contrastive == actual_prompt_contrastive
        expected_prompt = (
            "Is the implied meaning of the following responses yes or no:\n\n"
            "question: this is a test. \n"
            "response: do chicken have lips\n"
            "meaning: no\n\n"
        )
        test_prompt = ExamplePrompt(**self.test_prompt_two, contrastive=False)
        test_prompt_examples = [test_prompt._wrap_in_template(self.test_example_two, False, True)]
        assert expected_prompt == test_prompt._add_prompt_examples(test_prompt_examples)

    def test_prepare_prompt(self):
        random.seed(1)
        expected_prompt = (
            "Each following response to the question implies yes or no:\n\n"
            "question: Are you going to the party on Friday?\n"
            "response: Is the pope catholic?\n"
            "implicature: yes\n\n"
            "Does the following response to the question imply yes or no?\n\n"
            "question: this is a test. \n"
            "response: do chicken have lips\n"
            "implicature: no"
        )
        expected_prompt_contrastive = (
            "Each following response to the question implies yes or no:\n\n"
            "A: question: Are you going to the party on Friday?\n"
            "response: Is the pope catholic?\n"
            "implicature: yes\n"
            "B: question: Are you going to the party on Friday?\n"
            "response: Is the pope catholic?\n"
            "implicature: no\n"
            "Question: Which of the above texts is coherent, A or B?\nAnswer: A\n\n"
            "Does the following response to the question imply yes or no?\n\n"
            "A: question: this is a test. \n"
            "response: do chicken have lips\n"
            "implicature: no\n"
            "B: question: this is a test. \n"
            "response: do chicken have lips\n"
            "implicature: yes\n"
            "Question: Which of the above texts is coherent, A or B?\nAnswer: A"
        )
        test_prompt = ExamplePrompt(**self.test_prompt_one, contrastive=False)
        test_prompt_contrastive = ExamplePrompt(
            **self.test_prompt_one, contrastive=True
        )
        test_example = test_prompt._wrap_in_template(self.test_example_two, False, False)
        test_prompt_examples = [test_prompt._wrap_in_template(self.test_example_one, False, True)]
        test_example_contrastive = test_prompt_contrastive._wrap_in_template(
            self.test_example_two, False, False
        )
        test_prompt_examples_contrastive = [
            test_prompt_contrastive._wrap_in_template(self.test_example_one, False, True)
        ]
        actual_prompt_contrastive = test_prompt_contrastive._prepare_prompt(
            test_example_contrastive, test_prompt_examples_contrastive
        )
        assert expected_prompt == test_prompt._prepare_prompt(
            test_example, test_prompt_examples
        )
        assert expected_prompt_contrastive == actual_prompt_contrastive
        expected_prompt = (
            "Does the following response to the question imply yes or no?\n\n"
            "question: this is a test. \n"
            "response: do chicken have lips\n"
            "implicature: no"
        )
        assert expected_prompt == test_prompt._prepare_prompt(test_example)
        expected_output = (
            'The following are examples of the task:\n\nEsther asked "Have you found him yet?" and '
            'Juan responded "We\'re still looking", which means no.\n\nEsther asked "I feel horrible.'
            ' Debbie was furious that I lost her notes. Do you think I should apologize her again?" and Juan'
            ' responded "If I were you I would cool off for some days before I talk to her again.", which means no.'
            '\n\nEsther asked "I am going to conduct my psychology experiment this Saturday. I would be having a'
            ' handful. Would you help by writing the names of the participants." and Juan responded "I have got some'
            ' work on my own.", which means no.\n\nFinish the following sentence according to the task:\n\n'
            'Esther asked "do you want this so badly?" and Juan responded "It is everything.", which means no"'
        )
        test_prompt = ExamplePrompt(**self.test_prompt_three, contrastive=False)
        test_example = (
            'Esther asked "do you want this so badly?" and Juan responded "It is everything.", '
            'which means no"'
        )
        test_prompt_examples = [
            'Esther asked "Have you found him yet?" and Juan responded "We\'re still looking", which '
            "means no.",
            'Esther asked "I feel horrible. Debbie was furious that I lost her notes. Do you think I should'
            ' apologize her again?" and Juan responded "If I were you I would cool off for some days before'
            ' I talk to her again.", which means no.',
            'Esther asked "I am going to conduct my psychology experiment this Saturday. I would be having'
            ' a handful. Would you help by writing the names of the participants." and Juan responded "I'
            ' have got some work on my own.", which means no.',
        ]
        assert (
            test_prompt._prepare_prompt(test_example, test_prompt_examples)
            == expected_output
        )
        expected_output = (
            "The following are examples of the task:\n\nAnd this is also a test\n\nFinish the "
            "following sentence according to the task:\n\nThis is a test."
        )
        test_example = "This is a test."
        test_prompt_examples = ["And this is also a test"]
        assert (
            test_prompt._prepare_prompt(test_example, test_prompt_examples)
            == expected_output
        )
        expected_output = (
            "Finish the following sentence according to the task:\n\nThis is a test."
        )
        assert test_prompt._prepare_prompt(test_example, None) == expected_output
        expected_output = (
            'Esther asked Juan "Will these go away? These gas gauges?". '
            'Juan responded "It\'s probably a functional impairment.". '
            "The answer Juan implied by saying this was no."
        )
        test_prompt = ExamplePrompt(**self.test_prompt_four, contrastive=False)
        test_example = test_prompt._wrap_in_template(self.test_example_six, False, False)
        assert test_prompt._prepare_prompt(test_example) == expected_output

    def test_mask_template(self):
        random.seed(1)
        expected_prompt = (
            "question: this is a test. \n"
            "response: do chicken have lips\n"
            "implicature: [mask]"
        )
        test_prompt = ExamplePrompt(**self.test_prompt_one, contrastive=False)
        test_example = test_prompt._wrap_in_template(self.test_example_two, False, False)
        assert expected_prompt == test_prompt._mask_template(
            self.test_example_two["implicature"], test_example, "[mask]"
        )
        expected_prompt_contrastive = (
            "A: question: Are you going to the party on Friday?\n"
            "response: Is the pope catholic?\n"
            "implicature: yes\n"
            "B: question: Are you going to the party on Friday?\n"
            "response: Is the pope catholic?\n"
            "implicature: no\n"
            "Question: Which of the above texts is coherent, A or B?\nAnswer: [mask]"
        )
        test_prompt_contrastive = ExamplePrompt(
            **self.test_prompt_one, contrastive=True
        )
        test_example_contrastive = test_prompt_contrastive._wrap_in_template(
            self.test_example_one, False, False
        )
        actual_prompt_contrastive = test_prompt_contrastive._mask_template(
            self.test_example_two["implicature"], test_example_contrastive, "[mask]"
        )
        assert expected_prompt_contrastive == actual_prompt_contrastive
        expected_prompt_contrastive = (
            'A: Esther asked "this is a test. " and Juan responded "do chicken have lips", which means no.\n'
            'B: Esther asked "this is a test. " and Juan responded "do chicken have lips", which means yes.\n'
            "Question: Which of the above texts is coherent, A or B?\nAnswer: [mask]"
        )
        test_prompt_contrastive = ExamplePrompt(
            **self.test_prompt_three, contrastive=True
        )
        test_example_contrastive = test_prompt_contrastive._wrap_in_template(
            self.test_example_two, False, False
        )
        actual_prompt_contrastive = test_prompt_contrastive._mask_template(
            self.test_example_two["implicature"], test_example_contrastive, "[mask]"
        )
        assert expected_prompt_contrastive == actual_prompt_contrastive
        test_prompt = ExamplePrompt(**self.test_prompt_two, contrastive=False)
        test_example = test_prompt._wrap_in_template(self.test_example_two, False, False)
        expected_prompt = (
            "question: this is a test. \n"
            "response: do chicken have lips\n"
            "meaning: {{mask}}"
        )
        assert expected_prompt == test_prompt._mask_template(
            self.test_example_two["implicature"], test_example, "{{mask}}"
        )

    def test_prompt(self):
        random.seed(1)
        expected_prompt = (
            "Each following response to the question implies yes or no:\n\n"
            "question: Are you going to the party on Friday?\n"
            "response: Is the pope catholic?\n"
            "implicature: yes\n\n"
            "question: this is a test. \n"
            "response: do chicken have lips\n"
            "implicature: no\n\n"
            "Does the following response to the question imply yes or no?\n\n"
            "question: this is a test. \n"
            "response: do chicken have lips\n"
            "implicature: {{mask}}"
        )
        test_prompt = ExamplePrompt(**self.test_prompt_one, contrastive=False)
        assert (
            test_prompt.prompt(
                self.test_example_two,
                False,
                [self.test_example_one, self.test_example_two],
                "{{mask}}",
            )
            == expected_prompt
        )
        expected_prompt = (
            "Is the implied meaning of the following response yes or no:\n\n"
            "question: this is a test. \n"
            "response: do chicken have lips\n"
            "meaning: {{mask}}"
        )
        test_prompt = ExamplePrompt(**self.test_prompt_two, contrastive=False)
        assert (
            test_prompt.prompt(self.test_example_two, False, mask_token="{{mask}}")
            == expected_prompt
        )
        expected_prompt = (
            "Is the implied meaning of the following responses yes or no:\n\n"
            "question: this is a test. \n"
            "response: do chicken have lips\n"
            "meaning: no\n\n"
            "Is the implied meaning of the following response yes or no:\n\n"
            "question: Are you going to the party on Friday?\n"
            "response: Is the pope catholic?\n"
            "meaning: yes"
        )
        test_prompt = ExamplePrompt(**self.test_prompt_two, contrastive=False)
        assert (
            test_prompt.prompt(self.test_example_one, False, [self.test_example_two])
            == expected_prompt
        )
        expected_prompt_contrastive = (
            "The following are examples of the task:\n\n"
            'A: Esther asked "Are you going to the party on Friday?" and Juan responded "Is the pope catholic?", which means yes.\n'
            'B: Esther asked "Are you going to the party on Friday?" and Juan responded "Is the pope catholic?", which means no.\n'
            "Question: Which of the above texts is coherent, A or B?\nAnswer: A\n\n"
            "Finish the following sentence according to the task:\n\n"
            'A: Esther asked "this is a test. " and Juan responded "do chicken have lips", which means no.\n'
            'B: Esther asked "this is a test. " and Juan responded "do chicken have lips", which means yes.\n'
            "Question: Which of the above texts is coherent, A or B?\nAnswer: [mask]"
        )
        test_prompt_contrastive = ExamplePrompt(
            **self.test_prompt_three, contrastive=True
        )
        actual_prompt_contrastive = test_prompt_contrastive.prompt(
            self.test_example_two, False, [self.test_example_one], mask_token="[mask]"
        )
        assert actual_prompt_contrastive == expected_prompt_contrastive
        # Check prompt fails if prompt instruction not set but prompt examples are added.
        with pytest.raises(AssertionError):
            test_prompt = ExamplePrompt(**self.test_prompt_four, contrastive=False)
            test_prompt.prompt(self.test_example_one, False, [self.test_example_two])
