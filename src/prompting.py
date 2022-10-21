from typing import List, Dict
import random
import re
import os

from src.helpers import get_negative_binary_example


class Prompt:
    def prompt(
        self, example: Dict[str, str], is_false_example: bool, prompt_examples=None, mask_token=None
    ) -> str:
        raise NotImplementedError()


class ExamplePrompt(Prompt):
    """
    Holds prompt object that can take a template and form a prompt.

    Expects examples of the form:
    {
    "utterance": "Are you going to the party on Friday?",
    "response": "Is the pope catholic?",
    "implicature": "Yes."
    }

    Outputs examples of the form (depending on instantiation of properties):

    The following are examples of the task:

    Esther asked "Have you found him yet?" and Juan responded "We're still looking", which means no.

    Finish the following sentence according to the task:

    Esther asked "Are you going to the party on Friday?" and Juan responded "Is the pope catholic?", which means yes.

    """

    def __init__(
        self,
        example_template_text: str,
        contrastive: bool,
        prompt_instruction_text="",
        example_instruction_text="",
        phrase_to_put_mask="",
        **kwargs,
    ):
        """

        :param prompt_instruction_text:
        :param example_instruction_text:
        :param example_template_text:
        :param phrase_to_put_mask:
        """
        assert (
            example_template_text.count("%s") == 3
        ), "Need a template with 3 %s's to replace."
        self._contrastive = contrastive

        #
        self._previous_coherent_choices = None
        self._previous_coherent_choices_prompts = []

        self._prompt_instruction_text = prompt_instruction_text
        if self._prompt_instruction_text:
            self.prompt_instruction_set = True
        else:
            self.prompt_instruction_set = False
        self._example_instruction_text = example_instruction_text
        self._example_template_text = example_template_text
        self._phrase_to_put_mask = phrase_to_put_mask
        assert phrase_to_put_mask in example_template_text, (
            "phrase_to_put_mask(=%s) must be substring of"
            " example_template_text(=%s)." % (phrase_to_put_mask, example_template_text)
        )

    def _wrap_in_template(self, example: Dict[str, str], is_false_example: bool, prompt_example: bool):
        if not self._contrastive:
            return self._example_template_text % (
                example["utterance"],
                example["response"],
                example["implicature"],
            )
        else:
            contrastive_example = get_negative_binary_example(example)
            coherent_sentence = self._example_template_text % (
                example["utterance"],
                example["response"],
                example["implicature"],
            )
            incoherent_sentence = self._example_template_text % (
                contrastive_example["utterance"],
                contrastive_example["response"],
                contrastive_example["implicature"],
            )
            sentences = [coherent_sentence, incoherent_sentence]
            options = ["A", "B"]
            options_idx = [0, 1]
            if not prompt_example and not is_false_example:
                self._previous_coherent_choices = random.choice([0, 1])
                coherent_option = self._previous_coherent_choices
                self._previous_coherent_choices_prompts.clear()
                full_text_answer = coherent_sentence
            elif prompt_example and not is_false_example:
                coherent_option = random.choice([0, 1])
                self._previous_coherent_choices_prompts.append(coherent_option)
                full_text_answer = coherent_sentence
            elif prompt_example and is_false_example:
                coherent_option = self._previous_coherent_choices_prompts.pop(0)
                full_text_answer = coherent_sentence
            else:
                coherent_option = not self._previous_coherent_choices
                full_text_answer = coherent_sentence
            multiple_choice_answer = options[coherent_option]
            example = (
                f"A: {sentences[options_idx[coherent_option]]}\nB: {sentences[options_idx[not coherent_option]]}\n"
                f"Question: Which of the above texts is coherent, A or B?\nAnswer: {multiple_choice_answer}"
            )
            return example

    def _add_prompt_examples(self, prompt_example_texts: List[str]):
        assert len(prompt_example_texts), "Cannot prepare prompt without examples."
        template_text = self._prompt_instruction_text + "\n\n"
        k_shot_examples = "\n\n".join(prompt_example_texts)
        template_text = template_text + k_shot_examples + "\n\n"
        return template_text

    def _prepare_prompt(self, text_to_score: str, prompt_example_texts=None):
        template_text = ""
        if prompt_example_texts:
            template_text += self._add_prompt_examples(prompt_example_texts)
        if self._example_instruction_text:
            template_text += self._example_instruction_text + "\n\n"
        template_text += text_to_score
        return template_text

    def _mask_template(self, implicature: str, templated_text: str, mask_token: str):
        """Replaces implicature with mask token, e.g. if implicature='yes',
        templated_text='Is the pope catholic, meaning yes.'
        output is: ''Is the pope catholic, meaning [mask].''"""
        if not self._contrastive:
            assert (
                self._phrase_to_put_mask
            ), "Cannot mask example without phrase_to_put_mask set."
            return templated_text.replace(
                self._phrase_to_put_mask % implicature,
                self._phrase_to_put_mask % mask_token,
            )
        else:
            phrase_to_put_mask = "Answer: "
            templated_text_split = templated_text.split(phrase_to_put_mask)
            return templated_text_split[0] + phrase_to_put_mask + mask_token

    def prompt_for_completion(
        self, example: Dict[str, str], prompt_examples=None, mask_token=None
    ):
        if prompt_examples:
            assert self.prompt_instruction_set, (
                "Cannot add prompt examples with "
                "self._prompt_instruction_text=%s" % self._prompt_instruction_text
            )
        assert (
            self._phrase_to_put_mask
        ), "Cannot prompt for completion example without phrase_to_put_mask set."
        index_to_start_completion = self._phrase_to_put_mask.index(" %s")
        phrase_to_keep = self._phrase_to_put_mask[:index_to_start_completion]
        processed_example = self._wrap_in_template(example)
        index_to_start_completion = processed_example.index(phrase_to_keep) + len(
            phrase_to_keep
        )
        if prompt_examples:
            prompt_examples = [
                self._wrap_in_template(prompt_example)
                for prompt_example in prompt_examples
            ]
        processed_example = processed_example[:index_to_start_completion]
        if mask_token:
            processed_example = processed_example + f" {mask_token}"
        return self._prepare_prompt(processed_example, prompt_examples)

    def prompt(self, example: Dict[str, str], is_false_example: bool, prompt_examples=None, mask_token=None):
        if prompt_examples:
            assert self.prompt_instruction_set, (
                "Cannot add prompt examples with "
                "self._prompt_instruction_text=%s" % self._prompt_instruction_text
            )
        processed_example = self._wrap_in_template(example, is_false_example=is_false_example, prompt_example=False)
        if mask_token:
            processed_example = self._mask_template(
                example["implicature"], processed_example, mask_token
            )
        if prompt_examples:
            prompt_examples = [
                self._wrap_in_template(prompt_example, is_false_example=is_false_example, prompt_example=True)
                for prompt_example in prompt_examples
            ]
        return self._prepare_prompt(processed_example, prompt_examples)


def read_prompt_file(input_file: str) -> List[Dict[str, str]]:
    """
    Expects first line of file to have keys seperated by ; and each following,
    line to have values for those keys, seperated by ; and contained in apostrophes "..."
    :param input_file: path to file with prompts
    :return: each prompt is a dict with keys according to the first line, and values according to the following lines
    """
    assert os.path.exists
    prompts = []
    with open(input_file, "r") as infile:
        keys = []
        for i, row in enumerate(infile.readlines()):
            if i == 0:
                prompt_keys = [key.replace("\n", "").strip() for key in row.split(";")]
                keys.extend(prompt_keys)
            else:
                current_prompt = {}
                for j, value in enumerate(row.split(";")):
                    value = re.search('"(.*)"', value).group(1)
                    current_prompt[keys[j]] = value.replace(r"\n", "\n")
                prompts.append(current_prompt)
    return prompts
