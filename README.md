# Data-Driven Optimization of Ava’s Human-Written Prompts

## Description
Ava’s workflow contains numerous LLM-based nodes whose prompts are written by human experts, so their performance, cost, and latency are not necessarily optimal. Automated prompt-optimization techniques can refine these prompts. Among them, [OPRO](https://arxiv.org/abs/2309.03409) is both simple to implement and highly effective. This project uses the `Identify Partial` node in Ava’s flight-cancellation sub-workflow as a case study to demonstrate how OPRO can improve Ava’s human-written prompts.

## Environment Setup
```sh
# the env is based on python=3.13.5
# 1. create python virtual env
python -m venv .venv

# 2. activate the env
source .venv/bin/activate

# 3. install the requirements
pip install -r requirements.txt

# 4. create .env file with keys
OPENAI_API_KEY=
OPENAI_ORG_ID=
OPENAI_PROJECT_ID=

NEW_RELIC_API_KEY=
NEW_RELIC_ACCOUNT_ID=
```

## Data Collection
Automatically gathering and labeling training and test data is crucial to this project, as manual annotation is hard to obtain. Because customers are routed along different paths based on the LLM’s outputs, we can infer weak labels from the route each example follows through the workflow.

![Ava Workflow - Cancel not for all passengers (Full cancellation)](draw.io/Ava/cancel_not_for_all=False-PARTIAL-v2.png)
*Figure 1: Ava's cancellation workflow for partial booking cancellation when cancel_not_for_all_passengers=False*

![Ava Workflow - Cancel not for all passengers (Full cancellation)](draw.io/Ava/cancel_not_for_all=False-FULL-v2.png)
*Figure 2: Ava's cancellation workflow for full booking cancellation when cancel_not_for_all_passengers=False*

![Ava Workflow - Cancel not for all passengers (Full cancellation)](draw.io/Ava/cancel_not_for_all=True-null-v2.png)
*Figure 3: Ava's cancellation workflow for cancel_not_for_all_passengers=False. Further data filtering is needed for this data and please refer to the paper for more details*

Please refer to [Query](data/raw/logs/04222025-08182025/README.md) for the exact queires of collecting the data. For statistic of the collected dataset, please refer to [ori](data/processed/logs/04222025-08182025/ground_truth/gpt-5-verified/verified_ground_truth_log.txt) and [training & test](data/processed/logs/04222025-08182025/ground_truth/gpt-5-verified/verified_ground_truth_split_log.txt).

## How to run

```python
# 1. opro training
python -m src.ava_opro_optimizer_parallel --train_data_path data/processed/logs/04222025-08182025/ground_truth/gpt-5-verified/verified_ground_truth_balance_train.json --initial_prompt_file prompts/original/identify_partial.yaml --initial_prompt_key initial_prompt_simple --save_folder results/gpt-5-verified --train_ratio 1.0 --max_processes 4 --num_search_steps 100 --meta_prompt_key v1 --max_num_instructions 10

# 2. evaluation
python -m src.utils.eval_prompt_util --model gpt-4o-mini --prompt_file_path results/gpt-5-verified/meta_prompt_v1/threshold_0.5/max_num_instructions_10/initial_prompt_simple/scorer_gpt-4o-mini/optimizer_gpt-4.1/train_ratio_1.0/num_search_steps_100/num_gen_inst_4_num_exp_2_opt_temperature_1.0/optimized_prompt.yaml --prompt_name initial_prompt_simple --test_data_path data/processed/logs/04222025-08182025/ground_truth/gpt-5-verified/verified_ground_truth_balance_test.json --data_source gpt-5-verified --verbose --run_num 5
```









