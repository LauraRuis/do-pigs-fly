# OpenAI
# Davinci AKA GPT-3 175B
python -m src.probe_llm +experiment=particularised ++model_ids=openai-davinci ++k_shot=0 ++prompt_file=data/alignment_prompt_templates.csv
# text-davinci-001 AKA InstructGPT 175B
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci001 ++k_shot=0 ++prompt_file=data/alignment_prompt_templates.csv
# text-davinci-002 AKA InstructGPT unknown size
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci002 ++k_shot=0 ++prompt_file=data/alignment_prompt_templates.csv