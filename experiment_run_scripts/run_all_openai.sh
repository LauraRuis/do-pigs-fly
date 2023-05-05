# GPT-3
# Ada AKA GPT-3 350M
python -m src.probe_llm +experiment=particularised ++model_ids=openai-ada ++k_shot=0
python -m src.probe_llm +experiment=particularised ++model_ids=openai-ada ++k_shot=1
python -m src.probe_llm +experiment=particularised ++model_ids=openai-ada ++k_shot=5
python -m src.probe_llm +experiment=particularised ++model_ids=openai-ada ++k_shot=10
python -m src.probe_llm +experiment=particularised ++model_ids=openai-ada ++k_shot=15
python -m src.probe_llm +experiment=particularised ++model_ids=openai-ada ++k_shot=30
# Babbage AKA GPT-3 1.3b
python -m src.probe_llm +experiment=particularised ++model_ids=openai-babbage ++k_shot=0
python -m src.probe_llm +experiment=particularised ++model_ids=openai-babbage ++k_shot=1
python -m src.probe_llm +experiment=particularised ++model_ids=openai-babbage ++k_shot=5
python -m src.probe_llm +experiment=particularised ++model_ids=openai-babbage ++k_shot=10
python -m src.probe_llm +experiment=particularised ++model_ids=openai-babbage ++k_shot=15
python -m src.probe_llm +experiment=particularised ++model_ids=openai-babbage ++k_shot=30
# Curie AKA GPT-3 6.7B
python -m src.probe_llm +experiment=particularised ++model_ids=openai-curie ++k_shot=0
python -m src.probe_llm +experiment=particularised ++model_ids=openai-curie ++k_shot=1
python -m src.probe_llm +experiment=particularised ++model_ids=openai-curie ++k_shot=5
python -m src.probe_llm +experiment=particularised ++model_ids=openai-curie ++k_shot=10
python -m src.probe_llm +experiment=particularised ++model_ids=openai-curie ++k_shot=15
python -m src.probe_llm +experiment=particularised ++model_ids=openai-curie ++k_shot=30
# Davinci AKA GPT-3 175B
python -m src.probe_llm +experiment=particularised ++model_ids=openai-davinci ++k_shot=0
python -m src.probe_llm +experiment=particularised ++model_ids=openai-davinci ++k_shot=1
python -m src.probe_llm +experiment=particularised ++model_ids=openai-davinci ++k_shot=5
python -m src.probe_llm +experiment=particularised ++model_ids=openai-davinci ++k_shot=10
python -m src.probe_llm +experiment=particularised ++model_ids=openai-davinci ++k_shot=15
python -m src.probe_llm +experiment=particularised ++model_ids=openai-davinci ++k_shot=30

# InstructGPT
# Text-ada-001 AKA InstructGPT 350M
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textada001 ++k_shot=0
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textada001 ++k_shot=1
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textada001 ++k_shot=5
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textada001 ++k_shot=10
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textada001 ++k_shot=15
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textada001 ++k_shot=30
# text-babbage-001 AKA InstructGPT 1.3b
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textbabbage001 ++k_shot=0
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textbabbage001 ++k_shot=1
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textbabbage001 ++k_shot=5
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textbabbage001 ++k_shot=10
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textbabbage001 ++k_shot=15
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textbabbage001 ++k_shot=30
# text-curie-001 AKA InstructGPT 6.7B
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textcurie001 ++k_shot=0
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textcurie001 ++k_shot=1
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textcurie001 ++k_shot=5
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textcurie001 ++k_shot=10
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textcurie001 ++k_shot=15
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textcurie001 ++k_shot=30
# text-davinci-001 AKA InstructGPT 175B
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci001 ++k_shot=0
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci001 ++k_shot=1
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci001 ++k_shot=5
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci001 ++k_shot=10
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci001 ++k_shot=15
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci001 ++k_shot=30

# text-davinci-002 AKA InstructGPT unknown size
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci002 ++k_shot=0
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci002 ++k_shot=1
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci002 ++k_shot=5
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci002 ++k_shot=10
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci002 ++k_shot=15
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci002 ++k_shot=30

# text-davinci-003 AKA InstructGPT RLHF unknown size
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci003 ++k_shot=0
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci003 ++k_shot=1
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci003 ++k_shot=5
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci003 ++k_shot=10
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci003 ++k_shot=15
python -m src.probe_llm +experiment=particularised ++model_ids=openai-textdavinci003 ++k_shot=30