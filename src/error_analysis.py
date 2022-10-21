from collections import defaultdict, Counter
from transformers import AutoConfig
import matplotlib.pyplot as plt
import numpy as np
import json
import csv
import os


MODEL_COLORS = {'OpenAI': np.array([0.10588235, 0.61960784, 0.46666667, 1.        ]),
                'GPT-3': np.array([0.61960784, 0.10588235, 0.46666667, 1.        ]),
                'Davinci-002': np.array([0.85098039, 0.37254902, 0.00784314, 1.        ]),
                'Cohere': np.array( [0.45882353, 0.43921569, 0.70196078, 1.        ]),
                'OPT': np.array([0.90588235, 0.16078431, 0.54117647, 1.        ]),
                'EleutherAI': np.array([0.4       , 0.65098039, 0.11764706, 1.        ]),
                'BLOOM': np.array([0.90196078, 0.67058824, 0.00784314, 1.        ]),
                "BlenderBot": np.array([0.4       , 0.4       , 0.4       , 1.        ])}
PROMPT_GROUPING = {
    "prompt_template_1": "structured",
    "prompt_template_2": "natural",
    "prompt_template_3": "structured",
    "prompt_template_4": "structured",
    "prompt_template_5": "natural",
    "prompt_template_6": "natural"
}
GROUP_LINESTYLE = {
    "structured": "dashed",
    "natural": "dotted"
}
HUMAN_AVG_PERFORMANCE = 86.23333333333333
HUMAN_BEST_PERFORMANCE = 89.83333333333333
NUM_PARAMETERS = {
    "openai": {"text-ada-001": {"parameters": 350*10**6,
                                "d_model": 1024,
                                "num_layers": 24,
                                "d_attn": 16,
                                "d_ff": 0},
               "text-babbage-001": {"parameters": 1.3*10**9,
                                    "d_model": 2048,
                                    "num_layers": 24,
                                    "d_attn": 24,
                                    "d_ff": 0},
               "text-curie-001": {"parameters": 6.7*10**9,
                                  "d_model": 4096,
                                  "num_layers": 32,
                                  "d_attn": 32,
                                  "d_ff": 0},
               "text-davinci-001": {"parameters": 175*10**9,
                                    "d_model": 12288,
                                    "num_layers": 96,
                                    "d_attn": 96,
                                    "d_ff": 0},
               "objective": "LM",
               "training data size": 300*10**9,
               "compute": 0,
               "training data": "reddit outbound links with 3+ karma",
               "model display name": "OpenAI"
               },
    "gpt3": {"ada": {"parameters": 350*10**6,
                     "size_str": "350m",
                     "d_model": 1024,
                     "num_layers": 24,
                     "d_attn": 16,
                     "d_ff": 0},
             "babbage": {"parameters": 1.3*10**9,
                               "size_str": "1.3b",
                           "d_model": 2048,
                           "num_layers": 24,
                           "d_attn": 24,
                           "d_ff": 0},
               "curie": {"parameters": 6.7*10**9,
                             "size_str": "6.7b",
                         "d_model": 4096,
                         "num_layers": 32,
                         "d_attn": 32,
                         "d_ff": 0},
               "davinci": {"parameters": 175*10**9,
                               "size_str": "175b",
                            "d_model": 12288,
                            "num_layers": 96,
                            "d_attn": 96,
                            "d_ff": 0},
               "objective": "LM",
               "training data size": 300*10**9,
               "compute": 0,
               "training data": "reddit outbound links with 3+ karma",
               "model display name": "GPT-3"
               },
    "text-davinci-002": {"instructGPT": {"parameters": 175*10**9,
                                       "d_model": 0,
                                       "num_layers": 0,
                                       "d_attn": 0,
                                       "d_ff": 0},
                       "objective": "LM",
                       "training data size": 300*10**9,
                       "compute": 0,
                       "training data": "reddit outbound links with 3+ karma",
                       "model display name": "Davinci-002"
                       },
    "cohere": {"small": {"parameters": 409.3*10**6,
                         "nonembedding-parameters": 409.3*10**6 - 51463168,
                         "d_model": 0,
                         "num_layers": 0,
                         "d_attn": 0,
                         "d_ff": 0},
               "medium": {"parameters": 6.067*10**9,
                          "nonembedding-parameters": 6.067*10**9 - 205852672,
                          "d_model": 0,
                          "num_layers": 0,
                          "d_attn": 0,
                          "d_ff": 0},
               "large": {"parameters": 13.12*10**9,
                         "nonembedding-parameters": 13.12*10**9 - 257315840,
                         "d_model": 0,
                         "num_layers": 0,
                         "d_attn": 0,
                         "d_ff": 0},
               "xl": {"parameters": 52.4*10**9,
                      "nonembedding-parameters": 52.4*10**9 - 411705344,
                      "d_model": 0,
                      "num_layers": 0,
                      "d_attn": 0,
                      "d_ff": 0},
               "model display name": "Cohere"},
    "bigscience-bloom": {"176b": {"parameters": 176*10**9,
                                  "nonembedding-parameters": 172.6*10**9,
                                  "d_model": 14336,
                                  "num_layers": 70,
                                  "d_attn": 112,
                                  "d_ff": 0,
                                  "context_window": 2048},
                         "560m": {"parameters": 560 * 10 ** 6,
                                  "nonembedding-parameters": 302 * 10 ** 6,
                                  "d_model": 1024,
                                  "num_layers": 24,
                                  "d_attn": 16,
                                  "d_ff": 0,
                                  "context_window": 0},
                         "1b1": {"parameters": 1.1 * 10 ** 9,
                                 "nonembedding-parameters": 680 * 10 ** 6,
                                 "d_model": 0,
                                 "num_layers": 0,
                                 "d_attn": 0,
                                 "d_ff": 0,
                                 "context_window": 0},
                         "1b7": {"parameters": 1.7 * 10 ** 9,
                                 "nonembedding-parameters": 1.2 * 10 ** 9,
                                  "d_model": 0,
                                  "num_layers": 0,
                                  "d_attn": 0,
                                  "d_ff": 0,
                                  "context_window": 0},
                         "3b": {"parameters": 3 * 10 ** 9,
                                "nonembedding-parameters": 2.4 * 10 ** 9,
                                "d_model": 0,
                                "num_layers": 0,
                                "d_attn": 0,
                                "d_ff": 0,
                                "context_window": 0},
                         "7b1": {"parameters": 7.1 * 10 ** 9,
                                 "nonembedding-parameters": 6 * 10 ** 9,
                                  "d_model": 0,
                                  "num_layers": 0,
                                  "d_attn": 0,
                                  "d_ff": 0,
                                  "context_window": 0},
               "model display name": "BLOOM"
                         },
    "facebook-opt": {"6.7b": {"parameters": 6.7 * 10 ** 9,
                              "d_model": 0,
                              "num_layers": 0,
                              "d_attn": 0,
                              "d_ff": 0,
                              "context_window": 0},
                     "13b": {"parameters": 13 * 10 ** 9,
                              "d_model": 0,
                              "num_layers": 0,
                              "d_attn": 0,
                              "d_ff": 0,
                              "context_window": 0},
                     "30b": {"parameters": 30 * 10 ** 9,
                              "d_model": 0,
                              "num_layers": 0,
                              "d_attn": 0,
                              "d_ff": 0,
                              "context_window": 0},
                     "66b": {"parameters": 66 * 10 ** 9,
                              "d_model": 0,
                              "num_layers": 0,
                              "d_attn": 0,
                              "d_ff": 0,
                              "context_window": 0},
                     "175b": {"parameters": 175 * 10 ** 9,
                              "d_model": 12288,
                              "num_layers": 96,
                              "d_attn": 96,
                              "d_ff": 4*12288,
                              "context_window": 0},
                     "2.7b": {"parameters": 2.7 * 10 ** 9,
                              "d_model": 0,
                              "num_layers": 0,
                              "d_attn": 0,
                              "d_ff": 0,
                              "context_window": 0},
                     "1.3b": {"parameters": 1.3 * 10 ** 9,
                             "d_model": 0,
                             "num_layers": 0,
                             "d_attn": 0,
                             "d_ff": 0,
                             "context_window": 0},
                     "350m": {"parameters": 350 * 10 ** 6,
                              "nonembedding-parameters": 57 * 10 ** 6,
                              "d_model": 768,
                              "num_layers": 12,
                              "d_attn": 12,
                              "d_ff": 3072,
                              "context_window": 2048},
                     "125m": {"parameters": 125 * 10 ** 6,
                              "nonembedding-parameters": 0,
                              "d_model": 0,
                              "num_layers": 0,
                              "d_attn": 0,
                              "d_ff": 0,
                              "context_window": 0},
               "model display name": "OPT"
                     },
    "facebook-blenderbot": {"90m": {"parameters": 90 * 10 ** 6,
                              "d_model": 0,
                              "num_layers": 0,
                              "d_attn": 0,
                              "d_ff": 0,
                              "context_window": 0},
                     "3b": {"parameters": 3 * 10 ** 9,
                              "d_model": 0,
                              "num_layers": 0,
                              "d_attn": 0,
                              "d_ff": 0,
                              "context_window": 0},
                     "7b": {"parameters": 7 * 10 ** 9,
                              "d_model": 0,
                              "num_layers": 0,
                              "d_attn": 0,
                              "d_ff": 0,
                              "context_window": 0},
               "model display name": "BlenderBot"
                     },
    "EleutherAI-gpt-j": {"6b": {"parameters": 6 * 10 ** 9,
                                "d_model": 0,
                                "num_layers": 0,
                                "d_attn": 0,
                                "d_ff": 0,
                                "context_window": 0},
               "model display name": "EleutherAI"
                         },
    "EleutherAI-gpt-neo": {"20b": {"parameters": 20 * 10 ** 9,
                                   "d_model": 0,
                                   "num_layers": 0,
                                   "d_attn": 0,
                                   "d_ff": 0,
                                   "context_window": 0},
                           "125m": {"parameters": 125 * 10 ** 6,
                                    "d_model": 0,
                                    "num_layers": 0,
                                    "d_attn": 0,
                                    "d_ff": 0,
                                    "context_window": 0},
                           "1.3b": {"parameters": 1.3 * 10 ** 9,
                                    "d_model": 0,
                                    "num_layers": 0,
                                    "d_attn": 0,
                                    "d_ff": 0,
                                    "context_window": 0},
                           "2.7b": {"parameters": 2.7 * 10 ** 9,
                                    "d_model": 0,
                                    "num_layers": 0,
                                    "d_attn": 0,
                                    "d_ff": 0,
                                    "context_window": 0},
               "model display name": "EleutherAI"
                           },
    "EleutherAI-gpt-neox": {"20b": {"parameters": 20 * 10 ** 9,
                                   "d_model": 0,
                                   "num_layers": 0,
                                   "d_attn": 0,
                                   "d_ff": 0,
                                   "context_window": 0},
               "model display name": "EleutherAI"},
    "roberta": {"base": {"parameters": 125 * 10**6},
                "large": {"parameters": 355 * 10**6},
                "model display name": "RoBERTa"},
    "gpt2": {"medium": {"parameters": 354 * 10**6},
             "large": {"parameters": 774 * 10**6},
             "xl": {"parameters": 1.6 * 10**9},
             "model display name": "GPT-2"},
    "bert-base": {"cased": {"parameters": 110 * 10**6},
                  "uncased": {"parameters": 110 * 10**6},
                  "model display name": "BERT"},
}


def read_results_file_from_folder(results_folder):
    """Assumes file that starts with 'results' is the one to read. Only works if the folder has one file like this."""
    files_in_dir = [file for file in os.listdir(results_folder) if file.startswith("results")]
    assert len(files_in_dir) == 1, f"Found {len(files_in_dir)} result files in {results_folder} instead of 1."
    results_file = files_in_dir.pop()
    results_file = os.path.join(results_folder, results_file)
    with open(results_file, "r") as infile:
        results = json.load(infile)
    return results


def error_analysis_per_k(results_folder, output_folder):

    results = read_results_file_from_folder(results_folder)

    # Extract the models that are part of this results file and the k-shot argument.
    model_ids = list(results.keys())
    model_ids.remove("predictions")
    model_ids.remove("arguments")
    k_shot = str(results["arguments"]["k_shot"])

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Hardcode stuff
    translate_yes_to_no = {
        "yes": "no",
        "no": "yes"
    }
    template_keys_translation = {"prompt_template_0": "prompt_template_1",
                                 "prompt_template_1": "prompt_template_2",
                                 "prompt_template_2": "prompt_template_3",
                                 "prompt_template_3": "prompt_template_4",
                                 "prompt_template_4": "prompt_template_5",
                                 "prompt_template_5": "prompt_template_6"}

    # Loop over all the models and save results per k.
    result_analysis = {k_shot: {}}
    for model_id in model_ids:
        model_results_folder = os.path.join(output_folder, model_id)
        if not os.path.exists(model_results_folder):
            os.mkdir(model_results_folder)
        model_results_folder = os.path.join(output_folder, model_id, str(k_shot))
        if not os.path.exists(model_results_folder):
            os.mkdir(model_results_folder)

        # Extract the accuracy per prompt template
        template_results = {}
        for key, values in results[model_id].items():
            if "prompt_template_" in key:
                template_results[template_keys_translation[key]] = values["implicature_metrics"]

        # Save all the results
        model_results = {"mean_accuracy": results[model_id]["mean_accuracy"],
                         "std": results[model_id]["std"],
                         "template_results": template_results
                         }
        result_analysis[k_shot][model_id] = model_results
    example_results_per_model = {model_id: defaultdict(lambda: defaultdict(dict)) for model_id in model_ids}
    for i, predictions_per_model in enumerate(results["predictions"]):
        model_id = list(predictions_per_model.keys())[0]
        assert model_id in model_ids, f"Unknown model id {model_id}"
        model_results = predictions_per_model[model_id]
        original_example_id = i
        original_example = predictions_per_model["original_example"]
        prompt_examples = predictions_per_model["prompt_examples"]
        true = predictions_per_model["original_example"]["implicature"]
        for prompt_template in template_keys_translation.keys():
            example_correct = model_results[prompt_template]["implicature_result"]["example_correct"]
            pred = true if example_correct else translate_yes_to_no[true]
            example_results_per_model[model_id][template_keys_translation[prompt_template]][i] = {
                "id": original_example_id, "original_example": original_example, "true": true,
                "pred": pred, "correct": int(example_correct), "prompt_examples": prompt_examples}
    for model_id in example_results_per_model.keys():
        result_analysis[k_shot][model_id]["example_results"] = example_results_per_model[model_id]
    # Write results for all models to a file
    save_results(output_folder, result_analysis)


def save_results(save_folder, results):
    results_path = os.path.join(save_folder, "all_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as infile:
            existing_results = json.load(infile)
    else:
        existing_results = {}

    for k, results_per_k in results.items():
        if k not in existing_results:
            existing_results[k] = {}
        for model, model_results in results_per_k.items():
            if model in existing_results[k]:
                print(f"WARNING: overwriting results for {model} {k}-shot")
            existing_results[k][model] = model_results

    with open(results_path, "w") as outfile:
        json.dump(existing_results, outfile, indent=4)
    return results_path


def par_calc(d_model, num_layers, d_attn, d_ffn):
    return 2 * d_model * num_layers * (2 * d_attn + d_ffn)


def get_nonembedding_pars_s2s(hf_name):
    config = AutoConfig.from_pretrained(hf_name)
    d_model = config.hidden_size
    encoder_pars = par_calc(d_model, config.encoder_layers, config.encoder_attention_heads, config.encoder_ffn_dim)
    decoder_pars = par_calc(d_model, config.decoder_layers, config.decoder_attention_heads, config.decoder_ffn_dim)
    return encoder_pars + decoder_pars


def get_nonembedding_pars_openai(engine):
    d_model = NUM_PARAMETERS["openai"][engine]["d_model"]
    num_layers = NUM_PARAMETERS["openai"][engine]["num_layers"]
    d_attn = NUM_PARAMETERS["openai"][engine]["d_attn"]
    d_ffn = 4 * d_model
    return 2 * d_model * num_layers * (2 * d_attn + d_ffn)


def get_nonembedding_pars(hf_name):
    config = AutoConfig.from_pretrained(hf_name)
    if "blenderbot" in hf_name.lower():
        return get_nonembedding_pars_s2s(hf_name)
    if "opt-175b" in hf_name.lower():
        return get_nonembedding_pars_openai("text-davinci-001")
    d_model = config.hidden_size
    num_layers = config.num_hidden_layers
    d_attn = config.num_attention_heads
    if "gpt-j-6B" in hf_name:
        d_ffn = 16384
    elif "gpt-neo" in hf_name:
        d_ffn = 4 * d_model
    else:
        d_ffn = config.ffn_dim
    return 2 * d_model * num_layers * (2 * d_attn + d_ffn)


def parse_model_id(model_id: str):
    model = "-".join(model_id.split("-")[1:])
    if model == "text-davinci-002":
        model_class = "text-davinci-002"
        model = "InstructGPT"
    elif "text" in model:
        model_class = "openai"
    elif "cohere" in model_id:
        model_class = "cohere"
    else:
        model_class = "huggingface"
        model = model_id

    if "nonembedding-parameters" in NUM_PARAMETERS[model_class][model].keys():
        nonembedding_parameters = NUM_PARAMETERS[model_class][model]["nonembedding-parameters"]
    else:
        if model_class == "huggingface":
            nonembedding_parameters = get_nonembedding_pars(model)
        else:
            nonembedding_parameters = get_nonembedding_pars_openai(model)

    name = NUM_PARAMETERS[model_class]["model display name"]
    return nonembedding_parameters, name


def plot_scale_graph(results_path, models_to_show, label_order=None):
    with open(results_path, "r") as infile:
        results = json.load(infile)

    human_avg, human_best = HUMAN_AVG_PERFORMANCE, HUMAN_BEST_PERFORMANCE
    results_folder = results_path.split("/")[0]
    k_shot = [int(k) for k in list(results.keys())]
    lines = {}
    model_classes = []
    for k in k_shot:
        data = results[str(k)]
        if k not in lines:
            lines[k] = {}
        for model_id in data.keys():
            model_size, name = parse_model_id(model_id)
            if name not in models_to_show:
                continue
            if name not in lines[k]:
                lines[k][name] = {}
                lines[k][name]["mean_line"] = {"x": [], "y": [], "std": []}
            model_classes.append(name)
            mean_accuracy = data[model_id]["mean_accuracy"]
            std = data[model_id]["std"]
            lines[k][name]["mean_line"]["x"].append(model_size)
            lines[k][name]["mean_line"]["y"].append(mean_accuracy)
            lines[k][name]["mean_line"]["std"].append(std)
            lines[k][name]["name"] = name

    linewidth = 3
    markersize = 10
    model_classes = list(set(model_classes))

    # different plot per k-shot in subplots graph
    fig, axs = plt.subplots(3, 2, sharey=True, sharex=True, figsize=(16, 18), dpi=600)
    # model_colors = plt.cm.Dark2(np.linspace(0, 1, len(model_classes)))
    legend_lines, legend_labels = [], []
    x_min = float("inf")
    x_max = 0
    for i, k in enumerate(k_shot):
        ax = axs[int(i // 2), i % 2]
        for j, model_class in enumerate(list(model_classes)):
            color = MODEL_COLORS[model_class]
            x = lines[k][model_class]["mean_line"]["x"]
            name = lines[k][model_class]["name"]
            y = [y for _, y in sorted(zip(x, lines[k][model_class]["mean_line"]["y"]))]
            std = [x for _, x in sorted(zip(x, lines[k][model_class]["mean_line"]["std"]))]
            x = sorted(x)
            if x[0] < x_min:
                x_min = x[0]
            if x[-1] > x_max:
                x_max = x[-1]
            line = ax.errorbar(x, y,
                               yerr=std,
                               label=f"{name}-{k}-shot", marker="o", linestyle="dashed", color=color,
                               markersize=markersize, linewidth=linewidth)
            if k == 0:
                humanline = ax.hlines(y=human_avg, xmin=x_min, xmax=x_max, label="Avg. human", linestyles="solid",
                           color="grey", linewidth=linewidth)
                humanlinebest = ax.hlines(y=human_best, xmin=x_min, xmax=x_max, label="Best human",
                                      linestyles="solid",
                                      color="black", linewidth=linewidth)
            if k == 0:
                randomline = ax.hlines(y=50.0, xmin=x_min, xmax=x_max, label="Random chance", linestyles="dotted", color="red", linewidth=linewidth)
            if i == 0:
                legend_lines.append(line)
                legend_labels.append(f"{name}")
            # ax.set_xticks(sorted_x, sorted_x)  # Set text labels.
            plt.ylim(bottom=0., top=100.0)
            ax.set_xscale("log")
            ax.title.set_text(f"{k}-shot")
            ax.title.set_size(16)
            if i % 2 == 0:
                ylabel = "Accuracy (%)"
                ax.set_ylabel(ylabel, fontsize=24)
            ax.xaxis.set_tick_params(labelsize=18)
            ax.yaxis.set_tick_params(labelsize=20)
            if int(i // 2) == 2:
                ax.set_xlabel("Model Parameters (Non-Embedding)", fontsize=24)
    plt.ylim(bottom=0., top=100.0)
    legend_lines.append(humanline)
    legend_labels.append(f"Avg. human")
    legend_lines.append(humanlinebest)
    legend_labels.append(f"Best human")
    legend_lines.append(randomline)
    legend_labels.append(f"Random chance")
    fig.legend(legend_lines, legend_labels, fontsize=18)
    plt.suptitle(f"Performance on the implicature task \nfor all k-shot evaluations", fontsize=28)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(results_folder, f"accuracy_v_size_subplots.png"))
    plt.clf()

    # different plot per k-shot
    x_min = float("inf")
    x_max = 0
    k_to_str = {0: "Zero",
                1: "One",
                5: "Five",
                10: "Ten",
                15: "Fifteen",
                30: "Thirty"}

    for i, k in enumerate(k_shot):
        legend_lines, legend_labels = [], []
        legend_d = {}
        plt.figure(figsize=(16, 14), dpi=200)
        for j, model_class in enumerate(list(model_classes)):
            color = MODEL_COLORS[model_class]
            x = lines[k][model_class]["mean_line"]["x"]
            name = lines[k][model_class]["name"]
            y = [y for _, y in sorted(zip(x, lines[k][model_class]["mean_line"]["y"]))]
            std = [x for _, x in sorted(zip(x, lines[k][model_class]["mean_line"]["std"]))]
            x = sorted(x)
            if x[0] < x_min:
                x_min = x[0]
            if x[-1] > x_max:
                x_max = x[-1]
            line = plt.errorbar(x, y,
                                yerr=std,
                                label=f"{model_class}-{k}-shot", marker="o", linestyle="dashed", color=color,
                                markersize=markersize, linewidth=linewidth)
            if k == 0:
                humanline = plt.hlines(y=human_avg, xmin=x_min, xmax=x_max, label="Avg. human",
                                      linestyles="solid",
                                      color="grey", linewidth=linewidth)
                humanlinebest = plt.hlines(y=human_best, xmin=x_min, xmax=x_max,
                                          label="Best human",
                                          linestyles="solid",
                                          color="black", linewidth=linewidth)
            legend_lines.append(line)
            legend_labels.append(f"{name}")
            legend_d[f"{name}"] = line
        randomline = plt.hlines(y=50.0, xmin=x_min, xmax=x_max, label="Random chance", linestyles="dotted",
                                color="red", linewidth=linewidth)
        plt.ylim(bottom=0., top=100.0)
        plt.xscale("log")
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.xlabel("Model Parameters (Non-Embedding)", fontsize=32)
        ylabel = "Accuracy (%)"
        plt.ylabel(ylabel, fontsize=32)
        legend_lines.append(humanline)
        legend_labels.append(f"Avg. human")
        legend_d["Avg. human"] = humanline
        legend_lines.append(humanlinebest)
        legend_labels.append(f"Best human")
        legend_d["Best human"] = humanlinebest
        legend_lines.append(randomline)
        legend_labels.append(f"Random chance")
        legend_d["Random chance"] = randomline
        loc = "lower right"
        if label_order is not None:
            ordered_legend_lines = [legend_d[line_n] for line_n in label_order]
        else:
            ordered_legend_lines = [line for line in legend_d.values()]
        plt.legend(ordered_legend_lines, label_order, fontsize=28, loc=loc)
        plt.title(f"{k_to_str[k]}-shot accuracy on the implicature task.", fontsize=32)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1, top=0.9, right=0.9, left=0.1)
        plt.savefig(os.path.join(results_folder, f"accuracy_v_size_k={k}.png"))
        plt.clf()


def plot_all_lines(results_path, renormalize_metric=False, use_differences=True):
    linewidth = 3
    markersize = 10

    with open(results_path, "r") as infile:
        results = json.load(infile)
    results_folder = results_path.split("/")[0]
    k_shot = [int(k) for k in list(results.keys())]

    lines = {}
    model_ids = []
    model_ids_to_plot = ["openai-text-davinci-001",
                         "cohere-xl",]
    names = ["OpenAI", "Cohere"]
    classes = ["OpenAI", "Cohere"]
    sizes = ["175B", "52B"]
    zero_shot_per_model = {}
    zero_shot_per_model_per_template = defaultdict(dict)
    for k in k_shot:
        data = results[str(k)]
        for model_id in data.keys():
            model_size, name = parse_model_id(model_id)
            if model_id not in lines:
                lines[model_id] = {}
                lines[model_id]["mean_line"] = {"x": [], "y": [], "std": []}
                lines[model_id]["name"] = name
            model_ids.append(model_id)
            mean_accuracy = data[model_id]["mean_accuracy"]
            if k == 0:
                zero_shot_per_model[model_id] = mean_accuracy
            if use_differences:
                mean_accuracy = mean_accuracy - zero_shot_per_model[model_id]
            std = data[model_id]["std"]
            lines[model_id]["mean_line"]["x"].append(k)
            lines[model_id]["mean_line"]["y"].append(mean_accuracy)
            lines[model_id]["mean_line"]["std"].append(std)
            results_per_line = data[model_id]["template_results"]
            for i, template in enumerate(results_per_line.keys()):
                if i + 1 not in lines[model_id]:
                    lines[model_id][i + 1] = {"x": [], "y": []}
                lines[model_id][i + 1]["x"].append(k)
                y_result = results_per_line[template]
                if k == 0:
                    zero_shot_per_model_per_template[model_id][template] = y_result
                if not use_differences:
                    lines[model_id][i + 1]["y"].append(results_per_line[template])
                else:
                    lines[model_id][i + 1]["y"].append(results_per_line[template] - zero_shot_per_model_per_template[model_id][template])

    # different plot per model to plot with template breakdown
    fig, axs = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(18, 8), dpi=600)
    legend_lines, legend_labels = [], []
    for i, model_id in enumerate(list(model_ids_to_plot)):
        # ax = axs[int(i // 2), i % 2]
        ax = axs[i]
        for line_key, line_data in lines[model_id].items():
            if line_key != "mean_line" and line_key != "name":
                prompt_group = PROMPT_GROUPING[f"prompt_template_{line_key}"]
                linestyle = GROUP_LINESTYLE[prompt_group]
                line, = ax.plot(line_data["x"], line_data["y"], label=f"Prompt template {line_key}", marker="o",
                                linestyle=linestyle, markersize=markersize, linewidth=linewidth)
                if i == 1:
                    legend_lines.append(line)
                    legend_labels.append(f"Prompt template {line_key}")
        if i == 0:
            ax.legend(handles=legend_lines, labels=legend_labels, fontsize=17, loc="lower right")
        ax.set_xticks(k_shot, k_shot)
        ax.title.set_text(f"{names[i]}-{sizes[i]}")
        ax.title.set_size(16)
        if i == 0:
            ylabel = "Accuracy (%)"
            if renormalize_metric:
                ylabel = "Normalized " + ylabel
            if use_differences:
                ylabel = "Relative Accuracy (%)"
            ax.set_ylabel(ylabel, fontsize=24)
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=20)
        # if int(i // 2) == 2:
        #     ax.set_xlabel("k-shot", fontsize=24)
        ax.set_xlabel("k-shot", fontsize=24)
    ymin = 50.
    ymax = 100.
    if use_differences:
        ymin = -15
        ymax = 15
    plt.ylim(bottom=ymin, top=ymax)
    if not use_differences:
        plt.suptitle(f"Relative accuracy (w.r.t 0-shot) on the implicature task for the largest model of each model class \nfor all prompt templates.", fontsize=28)
    else:
        plt.suptitle(
            f"Relative accuracy (w.r.t. 0-shot) due to in-context examples for all prompt templates.",
            fontsize=24)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(results_folder, f"accuracy_v_k_subplots.png"))
    plt.clf()

    # Mean accuracy plot per model of accuracy v. k
    plt.figure(figsize=(16, 14), dpi=200)
    for i, model_id in enumerate(list(model_ids_to_plot)):
        name = names[i]
        color = MODEL_COLORS[classes[i]]
        if name == "OpenAI":
            zorder = 10
        else:
            zorder = None
        plt.errorbar(lines[model_id]["mean_line"]["x"], lines[model_id]["mean_line"]["y"],
                     yerr=lines[model_id]["mean_line"]["std"], color=color, zorder=zorder,
                     label=f"{lines[model_id]['name']}-{sizes[i]}", marker="o", linestyle="dashed",
                     markersize=markersize, linewidth=linewidth)
    if not renormalize_metric and not use_differences:
        plt.hlines(y=50.0, xmin=k_shot[0], xmax=k_shot[-1], label="Random chance", linestyles="dotted", color="red",
                   linewidth=linewidth)
    plt.legend(loc="lower right", fontsize=32)
    plt.xticks(k_shot, k_shot, fontsize=28)
    plt.yticks(fontsize=24)
    ymin = 50.
    ymax = 100.
    if use_differences:
        ymin = -10
        ymax = 10
    plt.ylim(bottom=ymin, top=ymax)
    plt.xlabel("In-context examples (k)", fontsize=32)
    plt.ylabel("Relative Accuracy (%)", fontsize=32)
    plt.subplots_adjust(bottom=0.1, top=0.9, right=0.9, left=0.1)
    if not use_differences:
        plt.title(f"Relative accuracy (w.r.t 0-shot) due to in-context examples.", fontsize=32)
    else:
        plt.title(f"Relative accuracy (w.r.t 0-shot) due to in-context examples.", fontsize=32)
    plt.savefig(os.path.join(results_folder, f"accuracy_v_k.png"))


def get_human_performance_all_examples(file_path: str):
  with open(file_path, "r") as infile:
    csv_reader = csv.DictReader(infile)
    annotators = defaultdict(list)
    ground_truth = []
    examples = []
    for row in csv_reader:
      for key, value in row.items():
        if "annotator" in key.lower():
          annotators[key].append(value.strip().lower())
        if "ground truth" in key.lower():
          ground_truth.append(value.strip().lower())
      example = {
          "source": "",
          "type": row["Ground truth"].strip().lower(),
          "utterance": row["Utterance"],
          "response": row["Response"],
          "implicature": row["Ground truth"].strip().lower()
      }
      examples.append(example)
  accuracies = []
  all_correct = [0] * 150
  for key, choices in annotators.items():
    correct = np.array(choices) == np.array(ground_truth)
    all_correct += correct
    accuracy = sum(correct) / len(correct) * 100.
    assert len(correct) == 150, "Wrong number of annotations."
    accuracies.append(accuracy)
  return accuracies, examples, all_correct


def get_all_human_performance(human_eval_files):
  all_correct_counts = []
  all_examples = []
  for human_eval_file in human_eval_files:
    accuracies, examples, all_correct = get_human_performance_all_examples(human_eval_file)
    all_correct_counts.extend(all_correct)
    all_examples.extend(examples)
  return all_examples, all_correct_counts


def extract_counter(results):
    k_shot = [0, 1, 5, 10, 15, 30]
    # Extract and group the accuracies per prompt template.
    # Get the average accuracy overall, grouping together different models and k
    template_scores_overall = defaultdict(list)
    # Get the accuracy per k-shot, grouping together different models
    template_scores_per_k = defaultdict(lambda: defaultdict(list))
    # Get the accuracy per model, grouping together k
    template_scores_per_model = defaultdict(lambda: defaultdict(list))
    # Get the accuracy per model per k, not grouping anything
    template_scores_per_model_per_k = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    example_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    models_sorted_by_size = defaultdict(list)
    model_sizes = defaultdict(list)
    model_to_group = {}
    for k in k_shot:
        for model, model_results in results[str(k)].items():
            if k == 0:
                model_size, name = parse_model_id(model)
                models_sorted_by_size[name].append(model)
                model_sizes[name].append(model_size)
                model_to_group[model] = name
            accuracy_per_template = model_results["template_results"]
            # template_rating = model_results["template_rating"]
            # likelihood_diff_per_template = model_results["likelihood_diffs_per_template"]
            for template in accuracy_per_template.keys():
                # Accuracy metrics
                template_scores_overall[template].append(accuracy_per_template[template])
                template_scores_per_k[k][template].append(accuracy_per_template[template])
                template_scores_per_model[model][template].append(accuracy_per_template[template])
                template_scores_per_model_per_k[model][k][template].append(accuracy_per_template[template])
                example_results[model][k][template] = model_results["example_results"][template]
        if k == 0:
            for model in models_sorted_by_size:
                models_sorted_by_size[model] = [x for _, x in
                                                sorted(zip(model_sizes[model], models_sorted_by_size[model]))]
                model_sizes[model] = sorted(model_sizes[model])
    examples_correct_count = Counter()
    example_data = {}
    counter_per_model_per_k = defaultdict(lambda: defaultdict(Counter))
    for model, examples_per_model in example_results.items():
        for k, examples_per_model_per_k in examples_per_model.items():
            for template, examples in examples_per_model_per_k.items():
                for example_id, ex_results in examples.items():
                    if example_id not in example_data:
                        cleaned_example = {}
                        for key, value in ex_results["original_example"].items():
                            cleaned_example[key] = value.replace("\r", "")
                        example_data[example_id] = cleaned_example
                    else:
                        cleaned_example = {}
                        for key, value in ex_results["original_example"].items():
                            cleaned_example[key] = value.replace("\r", "")
                        assert example_data[example_id] == cleaned_example
                    example_correct = ex_results["correct"]
                    if example_correct:
                        examples_correct_count[example_id] += 1
                        counter_per_model_per_k[model][k][example_id] += 1
                    else:
                        counter_per_model_per_k[model][k][example_id] += 0
    return counter_per_model_per_k, example_data, models_sorted_by_size, model_sizes


def type_label_analysis():
    file = "data/type_labels.csv"
    with open(file, "r") as infile:
        file_reader = csv.reader(infile)
        keys = []
        examples_with_type_labels = {}
        for i, row in enumerate(file_reader):
            if i == 0:
                keys.extend(row)
            else:
                example = {"utterance": row[0],
                           "response": row[1],
                           "implicature": row[2].strip(".").lower(),
                           "label": row[3],
                           "factual": row[4]}
                examples_with_type_labels[row[0].replace("\r", "")] = example

    project_folder = "error_analysis"
    results_file = os.path.join(project_folder, "all_results.json")
    with open(results_file, "r") as infile:
        results = json.load(infile)

    human_eval_files = ["data/human_eval/human_eval - 1-150.csv",
                        "data/human_eval/human_eval - 151-300.csv",
                        "data/human_eval/human_eval - 301-450.csv",
                        "data/human_eval/human_eval - 451-600.csv"]
    human_examples, human_counts = get_all_human_performance(human_eval_files)

    counter_per_model_per_k, example_data, models_sorted_by_size, model_sizes = extract_counter(results)
    k_shot = [0, 1, 5, 10, 15, 30]
    # Get the example IDs for the labeled examples
    examples_with_type_labels_id = {}
    found_labels = 0
    label_dist = Counter()

    for ex_id, example in example_data.items():
        key = example["utterance"].replace("\r", "")
        if key in examples_with_type_labels:
            example_labeled = examples_with_type_labels[example["utterance"].replace("\r", "")]
            found_labels += 1
        else:
            example_labeled = {"label": "Other"}
            example_labeled.update(example)
        label_dist[example_labeled["label"]] += 1
        label_dist["Mean"] += 1
        examples_with_type_labels_id[ex_id] = example_labeled

    # Let's select a view models to show this for, change below to view different models
    print(f"Options of model groups to show: {list(models_sorted_by_size.keys())}")
    models_to_show = ["Cohere", "OpenAI"]
    line_style_group = {"Cohere": "solid",
                        "OpenAI": "dashed"}
    for model in models_to_show:
        print(f"Will show {model}")
        print(f"-- with sizes: {models_sorted_by_size[model]}")

    labels = list(label_dist.keys())
    look_at_labels = ["World knowledge", "Idiom", "Rhetorical question", "Particularised", "Generalised", "Other"]
    num_labels = len(look_at_labels)

    lines = {}
    for group_name in models_to_show:
        lines[group_name] = {label: {"x": [], "y": [], "std": []} for label in look_at_labels}
        # = {"x": [], "y": [], "std": []}
        sorted_models = models_sorted_by_size[group_name]
        sorted_sizes = model_sizes[group_name]
        print(f"- Model group {group_name}")
        for i, model in enumerate(sorted_models):
            print(f"----- Model {model}")
            for k in k_shot:
                if k != 0:
                    continue
                print(f"---------------- {k}-shot")
                correct_per_type = Counter()
                for ex_id, example_labeled in examples_with_type_labels_id.items():
                    example_correct_count_zero_shot = counter_per_model_per_k[model][k][ex_id]
                    correct_per_type[example_labeled["label"]] += example_correct_count_zero_shot / 6
                    correct_per_type["Mean"] += example_correct_count_zero_shot / 6
                mean_correct = correct_per_type["Mean"]
                percentage_correct_mean = mean_correct / label_dist["Mean"] * 100
                for j, label_type in enumerate(look_at_labels):
                    type_correct = correct_per_type[label_type]
                    percentage_correct_label = type_correct / label_dist[label_type] * 100
                    lines[group_name][label_type]["y"].append(percentage_correct_label - percentage_correct_mean)
                    lines[group_name][label_type]["x"].append(sorted_sizes[i])
                    print(f"{label_type} absolute label mean: {percentage_correct_label}")
                    print(f"{label_type} absolute mean: {percentage_correct_mean}")

    print(f"- Model group humans")
    correct_per_type_humans = Counter()
    for ex_id, example_labeled in examples_with_type_labels_id.items():
        assert human_examples[int(ex_id)]["utterance"].replace("\r", "").replace("\n", "").strip() == example_data[ex_id]["utterance"].replace("\r", "").replace("\n", "").strip()
        example_correct_count_zero_shot = human_counts[int(ex_id)]
        correct_per_type_humans[example_labeled["label"]] += example_correct_count_zero_shot / 5
        correct_per_type_humans["Mean"] += example_correct_count_zero_shot / 5
    mean_correct_humans = correct_per_type_humans["Mean"]
    percentage_correct_mean_humans = mean_correct_humans / label_dist["Mean"] * 100
    for j, label_type in enumerate(look_at_labels):
        type_correct = correct_per_type_humans[label_type]
        percentage_correct_label = type_correct / label_dist[label_type] * 100
        # lines[group_name][label_type]["y"].append(percentage_correct_label - percentage_correct_mean_humans)
        # lines[group_name][label_type]["x"].append(sorted_sizes[i])
        print(f"{label_type} absolute label mean: {percentage_correct_label}")
        print(f"{label_type} absolute mean: {percentage_correct_mean_humans}")

    linewidth = 3
    markersize = 10

    colors = plt.cm.Dark2(np.linspace(0, 1, num_labels))
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(20, 10), dpi=200)
    legend_lines, legend_labels = [], []
    x_min = float("inf")
    x_max = 0
    for i, group_name in enumerate(models_to_show):
        # plt.figure(figsize=(16, 14), dpi=200)
        for j, (line_key, line_data) in enumerate(lines[group_name].items()):
            line, = ax[i].plot(line_data["x"], line_data["y"], label=f"Prompt template {line_key}", marker="o",
                               color=colors[j],
                               linestyle="solid", markersize=markersize, linewidth=linewidth)
            if min(line_data["x"]) < x_min:
                x_min = min(line_data["x"])
            if max(line_data["x"]) > x_max:
                x_max = max(line_data["x"])
            if i == 0:
                legend_lines.append(line)
                legend_labels.append(f"{line_key}")
        plt.xscale("log")
        ax[i].xaxis.set_tick_params(labelsize=24)
        ax[i].yaxis.set_tick_params(labelsize=24)
        ax[i].title.set_text(f"{group_name}")
        ax[i].title.set_size(24)
        if i == 0:
            ax[i].set_ylabel("Relative Accuracy (%)", fontsize=28)
        ax[i].set_xlabel("Model Parameters (Non-Embedding)", fontsize=28)
    for i in range(len(ax)):
        randomline = ax[i].hlines(y=0.0, xmin=x_min, xmax=x_max,
                                  label="Mean accuracy", linestyles="dotted", color="black", linewidth=linewidth)
    legend_lines.append(randomline)
    legend_labels.append("All types")
    ax[0].legend(legend_lines, legend_labels, fontsize=20, loc="lower left")
    plt.suptitle(f"Relative accuracy (w.r.t. mean) for each type of implicature.", fontsize=28)
    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    plt.savefig(f"error_analysis/type_labels_plot.jpg")

    print(f"Found {found_labels} non-other labeled examples")
    print(f"Distributed as: {label_dist}")
    print(f"Total count: {sum(label_dist.values()) - label_dist['Mean']}")


if __name__ == "__main__":
    # Gather all results per models and k and add it to one big file.
    models = ["cohere-small", "cohere-medium", "cohere-large", "cohere-xl",
              "openai-text-ada-001", "openai-text-babbage-001", "openai-text-curie-001", "openai-text-davinci-001"]
    k = [0, 1, 5, 10, 15, 30]
    for model in models:
        for k_shot in k:
            folder = f"results/{model}/{k_shot}-shot"
            error_analysis_per_k(folder, "error_analysis")

    file = "error_analysis/all_results.json"
    label_order = ["Best human", "Avg. human", "OpenAI", "Cohere", "Random chance"]
    models_to_show = ["OpenAI", "Cohere"]
    plot_scale_graph(file, models_to_show=models_to_show, label_order=label_order)
    plot_all_lines(file)
    type_label_analysis()
