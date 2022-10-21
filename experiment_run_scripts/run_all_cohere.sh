# Cohere
# Cohere small 409.3M
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-small ++k_shot=0
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-small ++k_shot=1
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-small ++k_shot=5
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-small ++k_shot=10
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-small ++k_shot=15
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-small ++k_shot=30
# Cohere medium 6.067B
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-medium ++k_shot=0
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-medium ++k_shot=1
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-medium ++k_shot=5
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-medium ++k_shot=10
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-medium ++k_shot=15
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-medium ++k_shot=30
# Cohere large 13.12B
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-large ++k_shot=0
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-large ++k_shot=1
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-large ++k_shot=5
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-large ++k_shot=10
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-large ++k_shot=15
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-large ++k_shot=30
# Cohere xl 52.4B
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-xl ++k_shot=0
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-xl ++k_shot=1
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-xl ++k_shot=5
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-xl ++k_shot=10
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-xl ++k_shot=15
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-xl ++k_shot=30