# Large language models are not zero-shot communicators
This repository concerns the <a href="TODO: paper link" target="_blank">paper</a> "Large language models are not zero-shot communicators". It contains the code for running evaluations with OpenAI's and Cohere's API's.
For evaluations on public HuggingFace models have a look at <a href="https://github.com/LauraRuis/lm-evaluation-harness" target="_blank">this repository</a>.

<a href="https://huggingface.co/datasets/UCL-DARK/ludwig" target="_blank">**HF Dataset**</a>

<a href="TODO: paper link" target="_blank">**Paper**</a>

<a href="https://lauraruis.github.io/2022/09/29/comm.html" target="_blank">**Blogpost**</a>

<a href="https://drive.google.com/file/d/1hWcuUpcNef0OGnbS_1PJmFR-LCoG2_oa/view?usp=sharing" target="_blank">**All results in a big file**</a>

**Research question**: To what extent do large language models understand conversational implicature?

<details close>
<summary><b>What is implicature?</b></summary>
Implicature is an aspect of language pragmatics and a crucial part of communication introduced by H.P. Grice in 1975 in his work "Logic and Conversation". Implicature is the act of meaning or implying one thing by saying something else. There's different types of implicatures, from simple ones like "Some guests came to the party" (implying not all guests came) to more complicated implicatures that depend on context like "A: Are you going to the party this Friday? B: There's a global pandemic." (implying no, or yes if A knows B is a reckless raver).
<br> <br>

**Some background** <br>
In his paper, Grice comes up with a set of maxims of conversation that we all adhere to, like "be relevant" and "do not say what you believe to be false". Grice says implicatures arise when these maxims are violated. For example, if A says "Smith doesn't seem to have a girlfriend these days", and B answers "He has been paying a lot of visits to New York lately"; unless B is violating the maxim of relevance, B is implying that Smith may have a girlfriend in New York.
</details>

## TOC

* [Install](#install)
* [Run evaluations](#run-evaluations)
* [Visualise results](#visualise-results)
* [Implementation details](#implementation-details)

## Install

Developed with Python 3.9.10.

Rust is a dependency for `transformers` library, install compiler with:

```bash
>> curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Other requirements:

```bash
>> python -m pip install -r requirements.txt
```

Note that to run any evaluations you need to have OpenAI and Cohere's API keys. Add these keys
to the two files in the folder `static` called `cohere_api_key.txt` and `openai_api_key.txt`. The former just needs
a single line with the key, the latter has the organization key on the first line and the API key on the second.

Check installation:

```bash
>> pytest tests/.
```

## Run evaluations

### Main 0-shot and k-shot experiments
The main experiments on OpenAI's models and Cohere's models are the zero-shot and k-shot experiments.
The models used for OpenAI are:
- GPT-3 (ada, babbage, curie, and davinci)
- InstructGPT (text-ada-001, text-babbage-001, text-curie-001, text-davinci-001)
- InstructGPT unknown size (text-davinci-002)

And for Cohere:
- small, medium, large, XL

Find the commands to run the OpenAI evaluations in `experiment_run_scripts/run_all_openai.sh` and Cohere in `experiment_run_scripts/run_all_cohere.sh`.

### Extra prompt templates
For the OpenAI models, we additionally used 3 prompt templates taken from the Sparrow paper. To run these experiments
run `experiment_run_scripts/run_extra_prompts_openai.sh`.

### Contrastive experiments
In the appendix, we did an experiment with a contrastive setup instead of a ranking setup. To run these experiments
run `experiment_run_scripts/run_contrastive.sh`. Note that this only runs the experiment for multiple choice options "A" and "B".
To change this like in the paper, adjust the code in `_wrap_in_template()` in `prompting.py`.

## Visualise results
For this section, unzip `results.zip`. It does not contain all the results in the paper, because those
are too large for GitHub, but it contains the results for InstructGPT and Cohere's models, and for OPT-175B.
If you want all results from the paper, find them <a href="https://drive.google.com/file/d/1hWcuUpcNef0OGnbS_1PJmFR-LCoG2_oa/view?usp=sharing" target="_blank">**here on drive**</a>.
To use that file, download it and place `all_results.json` in the folder `error_analysis`.
First unzip `results.zip`, then run:

```bash
>> python -m src.error_analysis
```

The individual results get added to a big file and the scale graphs are plot, the k-shot plots are made,
and the type label analysis is done. **Note that this can take several minutes.**

### Plot scale graph
After running the error analysis, the zero-shot scaling plot is in `error_analysis/accuracy_v_size_k=0.png`
You can adjust `models_to_show` and `label_order` at the bottom of `error_analysis.py` to change which models
to show on the plot.

### Plot k-shot results
After running the error analysis, the k-shot plots like the ones in the paper are
- `error_analysis/accuracy_v_k.png`
- `error_analysis/accuracy_v_k_subplots.png`


### Type label analysis
After running the error analysis as above, the type labels accuracies are printed
and the plot is `error_analysis/type_labels_plot.png`
