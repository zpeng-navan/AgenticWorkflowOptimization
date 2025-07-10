# AgenticWorkflowOptimization
Internship project to optimize Ava's workflow

## Environment Setup
```sh
# in vscode cmd+shift+p, choose the following options in order
# Python: Select Python Interpret
# Create Virtual Environment
# .venv
# to create python virutal environment
# 1. install pytorch
pip3 install torch torchvision torchaudio
# 2. install TextGrad
pip install textgrad[vllm]
# 3. install Trace
# 4. install DSPy
```

## Data Collection 

### New Relic Script
Customers' logs can be mined from `New Relic` with the following quereis:

```sh
# to filter all the LLM's inputs and outputs
applicationName:"ml-flow-svc" environment:"prod" NODE_ID:"vsat41bgk-lyyx97j8" ("prompt:" OR "completion:")
# to filter certain nodes' info
applicationName:"ml-flow-svc" environment:"prod" NODE_ID:"mr32xen3g-lxoztnxo"
```
Each session has an unique id `CHANNEL_ID` which can be utilized to extract the interactions for certain session. Once interactions are extracted, use `time` to sort them by order. 

New Relic can only download the recent 5000 rows of logs.

### Jun 1, 12:00 AM to Jun 14, 12:00 AM
Use this time range `Jun 1, 12:00 am to Jun 14, 12:00 am (PDT)` to extract the following information.

1. for LLM-based node, only extract the prompts and responses
2. for first node from which the workflow starts and the last node at which the workflow ends, only the `ENTER_NODE` and `EXIT_NODE` messages are extracted
3. for other nodes, export all the messages





