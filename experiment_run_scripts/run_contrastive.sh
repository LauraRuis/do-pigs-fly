# text-davinci-002 AKA InstructGPT unknown size
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci002 ++k_shot=0 ++task=contrastive ++prompt_file=data/two_prompt_templates.txt
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci002 ++k_shot=1 ++task=contrastive ++prompt_file=data/two_prompt_templates.txt
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci002 ++k_shot=5 ++task=contrastive ++prompt_file=data/two_prompt_templates.txt