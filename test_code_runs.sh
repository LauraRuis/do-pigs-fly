#! /bin/bash
echo "This script should not take more than a minute to run."
echo "Note that an OpenAI API key and organization ID needs to be set to run this in static/openai_api_key.txt. The costs are neglible because we use the cheapest model for only 5 test examples."
python -m src.probe_llm +experiment=particularised ++seed=1 ++model_ids=openai-ada ++objectives=lm ++max_num_evaluations=5 &> temp.txt
rm raw.csv
results_file=$(cat temp.txt | sed -n -e 's/^.*results to: //p')
data_file=$(cat temp.txt | sed -n -e 's/^.*data to: //p')
accuracy=$(cat "${results_file}" | sed -n -e 's/^.*        "mean_accuracy": //p')
result=$(awk '{print $1}' <<<"${accuracy} ")
if [[ $result < 60 ]] && [[ $result > 50 ]];
then
  echo "PASSED"
  rm "${results_file}"
  rm "${data_file}_implicature.json"
  rm temp.txt
  rm probe_llm.log
  rm -r results
else
  echo "FAILED. Required accuracy to be roughly ~58.3 but got ${accuracy}. Check temp.txt or probe_llm.log for clues on what went wrong. If there is no in temp.txt error, it might be that OpenAI's engine ada has changed or there is more stochasticity in the API."
fi