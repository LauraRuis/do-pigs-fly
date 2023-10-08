# gpt-3.5-turbo AKA ChatGPT RLHF unknown size
python -m src.probe_llm +experiment=particularised ++model_ids=openai-chatgpt ++k_shot=0 ++task=completion ++prompt_file=data/prompt_templates_completion.txt
python -m src.probe_llm +experiment=particularised ++model_ids=openai-chatgpt ++k_shot=1 ++task=completion ++prompt_file=data/prompt_templates_completion.txt
python -m src.probe_llm +experiment=particularised ++model_ids=openai-chatgpt ++k_shot=5 ++task=completion ++prompt_file=data/prompt_templates_completion.txt
python -m src.probe_llm +experiment=particularised ++model_ids=openai-chatgpt ++k_shot=10 ++task=completion ++prompt_file=data/prompt_templates_completion.txt
python -m src.probe_llm +experiment=particularised ++model_ids=openai-chatgpt ++k_shot=15 ++task=completion ++prompt_file=data/prompt_templates_completion.txt
python -m src.probe_llm +experiment=particularised ++model_ids=openai-chatgpt ++k_shot=30 ++task=completion ++prompt_file=data/prompt_templates_completion.txt

# gpt-3.5-turbo AKA ChatGPT RLHF unknown size
python -m src.probe_llm +experiment=particularised ++model_ids=openai-gpt4 ++k_shot=0 ++task=completion ++prompt_file=data/prompt_templates_completion.txt
python -m src.probe_llm +experiment=particularised ++model_ids=openai-gpt4 ++k_shot=1 ++task=completion ++prompt_file=data/prompt_templates_completion.txt
python -m src.probe_llm +experiment=particularised ++model_ids=openai-gpt4 ++k_shot=5 ++task=completion ++prompt_file=data/prompt_templates_completion.txt
python -m src.probe_llm +experiment=particularised ++model_ids=openai-gpt4 ++k_shot=10 ++task=completion ++prompt_file=data/prompt_templates_completion.txt
python -m src.probe_llm +experiment=particularised ++model_ids=openai-gpt4 ++k_shot=15 ++task=completion ++prompt_file=data/prompt_templates_completion.txt
python -m src.probe_llm +experiment=particularised ++model_ids=openai-gpt4 ++k_shot=30 ++task=completion ++prompt_file=data/prompt_templates_completion.txt