# 1-shot and 5-shot for text-davinci-001 and Cohere-command-xl with random labels for the prompt examples
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci001 ++k_shot=1 ++random_labels=true
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci001 ++k_shot=5 ++random_labels=true
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-commandxl ++k_shot=1 ++random_labels=true
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-commandxl ++k_shot=5 ++random_labels=true