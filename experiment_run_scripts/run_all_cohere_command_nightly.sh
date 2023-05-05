# Cohere
# Cohere medium 6.067B
#python -m src.probe_llm +experiment=particularised ++model_ids=cohere-commandmediumnightly ++k_shot=0
#python -m src.probe_llm +experiment=particularised ++model_ids=cohere-commandmediumnightly ++k_shot=1
#python -m src.probe_llm +experiment=particularised ++model_ids=cohere-commandmediumnightly ++k_shot=5
#python -m src.probe_llm +experiment=particularised ++model_ids=cohere-commandmediumnightly ++k_shot=10
#python -m src.probe_llm +experiment=particularised ++model_ids=cohere-commandmediumnightly ++k_shot=15
#python -m src.probe_llm +experiment=particularised ++model_ids=cohere-commandmediumnightly ++k_shot=30
# Cohere xl 52.4B
#python -m src.probe_llm +experiment=particularised ++model_ids=cohere-commandxl ++k_shot=0
#python -m src.probe_llm +experiment=particularised ++model_ids=cohere-commandxlnightly ++k_shot=1
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-commandxlnightly ++k_shot=5
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-commandxlnightly ++k_shot=10
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-commandxlnightly ++k_shot=15
python -m src.probe_llm +experiment=particularised ++model_ids=cohere-commandxlnightly ++k_shot=30