import pytest

from src.dataset import (
    ImplicatureSuperDataset,
    ParticularisedImplicatureDataset,
)

TEST_DATA = {
    "class": "generalized conversational implicature",
    "test_data": [
        {
            "source": "pastor_cerezuela_et_al_2018",
            "type": "Q",
            "utterance": "Yesterday Juan tried to jump over the wall.",
            "meaning": "Juan did not manage to jump over the wall.",
            "options": [
                "Juan did not manage to jump over the wall.",
                "Juan did manage to jump over the wall.",
                "Juan managed to stain his pants with mud.",
            ],
        },
        {
            "source": "pastor_cerezuela_et_al_2018",
            "type": "Q",
            "utterance": "Some guests came to Maria’s party.",
            "meaning": "Not all the guests came that Maria expected.",
            "options": [
                "Everyone Maria invited came.",
                "Not all the guests came that Maria expected.",
                "Exactly three guests came.",
            ],
        },
        {
            "source": "pastor_cerezuela_et_al_2018",
            "type": "I",
            "utterance": "If Juan does not do his homework, his mother punishes him. Juan’s mother punished him.",
            "meaning": "Juan’s mother punished him because he didn’t do his homework.",
            "options": [
                "Juan’s mother punished him because he did something wrong.",
                "Juan’s mother punished him because he didn’t do his homework.",
                "Juan’s mother loves her son a lot.",
            ],
        },
    ],
    "dev_data": [
        {
            "source": "pastor_cerezuela_et_al_2018",
            "type": "I",
            "utterance": "Maria and Juan, a happy married couple, bought a new flat.",
            "meaning": "Maria and Juan bought one flat to live in together.",
            "options": [
                "Maria and Juan bought one flat to live in together.",
                "Maria bought one flat, and Juan bought another one.",
                "Juan wants to leave Maria and buy a flat far away from her.",
            ],
        },
        {
            "source": "pastor_cerezuela_et_al_2018",
            "type": "M",
            "utterance": "It is not impossible for the teacher to arrive late.",
            "meaning": "Although it doesn’t happen very often, the teacher could arrive late.",
            "options": [
                "Although it doesn’t happen very often, the teacher could arrive late.",
                "It is quite possible that the teacher will arrive late.",
                "It is very common for the teacher to arrive late. ",
            ],
        },
    ],
}


def test_empty_fails() -> None:
    dataset = ImplicatureSuperDataset()
    assert dataset._data is None, "Dataset not initialised empty."
    with pytest.raises(Exception):
        dataset.print_example()
        dataset.read_statistics()


def test_set_data() -> None:
    dataset = ImplicatureSuperDataset()
    dataset.set_data(TEST_DATA)
    expected_num_test_examples = 3
    expected_num_dev_examples = 2
    assert len(dataset._data["test_data"]) == expected_num_test_examples
    assert len(dataset._data["dev_data"]) == expected_num_dev_examples


def test_read_statistics() -> None:
    dataset = ImplicatureSuperDataset()
    dataset.set_data(TEST_DATA)
    expected_num_examples = 3
    expected_per_source = {"pastor_cerezuela_et_al_2018": 3}
    expected_per_type = {"Q": 2, "I": 1, "M": 0}
    data_statistics = dataset.get_statistics()
    assert data_statistics["test"]["num_examples"] == expected_num_examples
    assert data_statistics["test"]["examples_per_source"] == expected_per_source
    assert data_statistics["test"]["Q"] == expected_per_type["Q"]
    assert data_statistics["test"]["I"] == expected_per_type["I"]
    assert data_statistics["test"]["M"] == expected_per_type["M"]


def test_update_statistics() -> None:
    dataset = ImplicatureSuperDataset()
    dataset.set_data(TEST_DATA)
    extra_example = {
        "source": "test_source_1",
        "type": "M",
        "utterance": "It is not impossible for the teacher to arrive late.",
        "meaning": "Although it doesn’t happen very often, the teacher could arrive late.",
        "options": [
            "Although it doesn’t happen very often, the teacher could arrive late.",
            "It is quite possible that the teacher will arrive late.",
            "It is very common for the teacher to arrive late. ",
        ],
    }
    dataset.add_example(extra_example, "dev")
    expected_num_examples = 3
    expected_per_source = {"pastor_cerezuela_et_al_2018": 2, "test_source_1": 1}
    expected_per_type = {"Q": 0, "I": 1, "M": 2}
    data_statistics = dataset.get_statistics()
    assert data_statistics["dev"]["num_examples"] == expected_num_examples
    assert data_statistics["dev"]["examples_per_source"] == expected_per_source
    assert data_statistics["dev"]["Q"] == expected_per_type["Q"]
    assert data_statistics["dev"]["I"] == expected_per_type["I"]
    assert data_statistics["dev"]["M"] == expected_per_type["M"]


def test_parse_example():
    dataset = ImplicatureSuperDataset()
    dataset.set_data(TEST_DATA)
    actual_example, actual_prompt_examples = dataset._parse_example(i=0, k_shot=2)
    actual_prompt_examples_meanings = [ex["meaning"] for ex in actual_prompt_examples]
    expected_example = TEST_DATA["test_data"][0]
    expected_prompt_examples = [TEST_DATA["dev_data"][0], TEST_DATA["dev_data"][1]]
    assert actual_example == expected_example
    assert expected_prompt_examples[0]["meaning"] in actual_prompt_examples_meanings
    assert expected_prompt_examples[1]["meaning"] in actual_prompt_examples_meanings
    # Also test for example that is at end of dataset, such that the expected prompt examples start again
    # at the first indices
    actual_example, actual_prompt_examples = dataset._parse_example(i=2, k_shot=1)
    actual_prompt_examples_meanings = [ex["meaning"] for ex in actual_prompt_examples]
    expected_example = TEST_DATA["test_data"][2]
    expected_prompt_examples = [TEST_DATA["dev_data"][0], TEST_DATA["dev_data"][1]]
    assert actual_example == expected_example
    assert (
        expected_prompt_examples[0]["meaning"] in actual_prompt_examples_meanings
        or expected_prompt_examples[1]["meaning"] in actual_prompt_examples_meanings
    )


def test_get_implicature_iterator():
    dataset = ImplicatureSuperDataset()
    data = {
        "class": "generalized conversational implicature",
        "test_data": [
            {
                "source": "pastor_cerezuela_et_al_2018",
                "type": "M",
                "utterance": "It is not impossible for the teacher to arrive late.",
                "meaning": "Although it doesn’t happen very often, the teacher could arrive late.",
                "options": [
                    "Although it doesn’t happen very often, the teacher could arrive late.",
                    "It is quite possible that the teacher will arrive late.",
                    "It is very common for the teacher to arrive late. ",
                ],
            }
        ],
        "dev_data": [
            {
                "source": "pastor_cerezuela_et_al_2018",
                "type": "Q",
                "utterance": "Some guests came to Maria’s party.",
                "meaning": "Not all the guests came that Maria expected.",
                "options": [
                    "Everyone Maria invited came.",
                    "Not all the guests came that Maria expected.",
                    "Exactly three guests came.",
                ],
            }
        ],
    }
    example_one = data["test_data"][0]
    example_two = data["dev_data"][0]
    dataset.set_data(data)
    iterator = dataset.get_implicature_iterator(k_shot=1)

    # Due to randomness of iterator, we can either get the first example as the example and the other as the prompt...
    # ...or vice-versa.
    for i, actual_output in enumerate(iterator):
        if i == 0:
            actual_example = actual_output["example"]
            actual_prompts = actual_output["prompts"][0]
            if actual_example == example_one:
                assert actual_prompts == example_two
            elif actual_example == example_two:
                assert actual_prompts == example_one
            else:
                raise AssertionError()


@pytest.mark.parametrize(
    "test_input, expected_output",
    [
        (
            {
                "Context utterance": "I keep putting off getting my passport application.\n\n",
                "Response utterance": "Thank Goodness I didn't drag my feet on that one.",
                "Implicature": "I already sent my passport application",
            },
            None,
        ),
        (
            {
                "Context utterance": "I keep putting off getting my passport application.\n\n",
                "Response utterance": "Thank Goodness I didn't drag my feet on that one.",
                "Implicature": "No. I went to the car.",
            },
            {
                "source": "",
                "type": "no",
                "utterance": "I keep putting off getting my passport application.",
                "response": "Thank Goodness I didn't drag my feet on that one.",
                "implicature": "no",
            },
        ),
        (
            {
                "Context utterance": "I keep putting off getting my passport application.\n\n",
                "Response utterance": "Thank Goodness I didn't drag my feet on that one.",
                "Implicature": "I did not go to the car.",
            },
            None,
        ),
        (
            {
                "Context utterance": "I keep putting off getting my passport application.\n\n",
                "Response utterance": "Thank Goodness I didn't drag my feet on that one.",
                "Implicature": "I did not go yes to the car.",
            },
            None,
        ),
        (
            {
                "Context utterance": "I keep putting off getting my passport application.\n\n",
                "Response utterance": "Thank Goodness I didn't drag my feet on that one.",
                "Implicature": "Yes I did go to the car.",
            },
            {
                "source": "",
                "type": "yes",
                "utterance": "I keep putting off getting my passport application.",
                "response": "Thank Goodness I didn't drag my feet on that one.",
                "implicature": "yes",
            },
        ),
    ],
)
def test_filter_examples_no_control(test_input, expected_output):
    dataset = ParticularisedImplicatureDataset()
    assert dataset._filter_examples(test_input) == expected_output
