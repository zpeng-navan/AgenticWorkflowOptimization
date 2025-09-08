# Database

## dwh.data_warehouse.ava_chat_history_api_raw_data
contains the chat history: customer with Ava, and **customer with human agent**

## dwh.base_dbt_dwh.base_ava_ml_flow_node
node-level grain. Contains the attributes of a node

## dwh.data_warehouse.fact_ava_flow
product-defined flow-level grain. Flow is the sub workflow name, like `cancel_booking`. It has important columns:

`flow`: the name of the flow, like `cancel_booking`
`channel_sid`: the unique id of each session, launched by the customer when customers talk with Ava
`outcome`: the exit status of this workflow
`exit_code_array`: a list of nodes visited in order when going through this flow
important signals of the current flow, for instance, `cancel_booking` has `task`, `domain`, `product`, etc

Questions:
what do `is_booking_changeable` and `is_booking_cancellable` represent?

# Flight Cancellation Status
## agent
exit code: USER_ASKED_FOR_AGENT

Only `confirm leg` flows into `agent`. The conversation ending with `agent` may have the following reasons:
+ the customer prefers a human agent and directly ask for a human agent
+ Ava cannot solve the customer's problem and the customer asks for human agent
+ + the label `partial_or_full` is incorrectly predicted by `identify partial`
+ + other issues

## Show CAST
exit code: SUCCESS

The flight cancellation is processed successfully by Ava's automatic workflow. We can confidently say the two labels `partial_or_full` and `cancel_not_for_all_passengers` are correct and should be the ground truth labels. However, `cancel_not_for_all_passengers=False` for all `SUCCESS` and there is no `cancel_not_for_all_passengers=True` conversation in this scenario. In this scenario, there are more `partial_or_full=FULL` than `partial_or_full=PARTIAL`.

## identify partial
### cancel_not_for_all_passengers=True
```
  return {
    exitCode: 'AVA_AGENT',
    outcome: 'AGENT',
    reason: 'TASK_NOT_AUTOMATED',
    thinking: 'Cant cancel boooking for fewer than all passengers in the booking'
  };
```
We can filter the conversation end with thinking: 'Cant cancel boooking for fewer than all passengers in the booking' to get data with label `cancel_not_for_all_passengers=True`. This label should be true postive if there is not following up conversations. Otherwise, it may be false postive. The conversation filtered by this method have no `partial_or_full` label. Thus, we should treat the prediction of `partial_or_full` and `cancel_not_for_all_passengers` separately. 

## Confirm cancel
```
addSummary(`User thought about cancelling the booking ${PROJECT.booking.bookingId}, and decided not to. Probably still need help about this booking.`);
  gotoLink('exit');
  return {
    exitCode: 'EXIT',
    outcome: 'EXIT',
    reason: 'NOP',
    thinking: 'User decided not proceed with cancellation'
  };
```
Usually, the customer should proceed to cancel the flight booking, however, the customer stops and one reason may be that the `partial_or_full` and `cancel_not_for_all_passengers` are predicted incorrectly by `identify partial`.

# Dataset
Based on our analysis of the flight-cancellation workflow, we can build two datasets to drive further improvements:

1. A ground-truth dataset for prompt optimization.  
2. A failure (hard-case) dataset for workflow debugging and additional prompt refinement.

Before using the failure dataset for prompt optimization, we need to clean it by only picking up the cases whose failure reason is "labels incorrectly predicted by identify partial".

## Ground Truth Dataset
The ground truth dataset consists of the following two sources:

1. Use exit code `SUCCESS` to collect all the conversations successfully handled by Ava's workflow. The two labels `partial_or_full` and `cancel_not_for_all_passengers` are correct and should be the ground truth labels. However, `cancel_not_for_all_passengers=False` for all `SUCCESS` and there is no `cancel_not_for_all_passengers=True` conversation in this scenario. In addition, in this scenario, there are more `partial_or_full=FULL` than `partial_or_full=PARTIAL`. 2.8K sessions (04222025-08182025)

2. Use exit code:
    ```
    return {
    exitCode: 'AVA_AGENT',
    outcome: 'AGENT',
    reason: 'TASK_NOT_AUTOMATED',
    thinking: 'Cant cancel boooking for fewer than all passengers in the booking'
    };
    ```
    to get data with label `cancel_not_for_all_passengers=True`. 241 sessions (04222025-08182025) 
    
`LIMITATION`: 

This label of the second data source should be true postive if there is not following up conversations. Otherwise, it may be false postive. 

`IMPORTANT`:

The conversations of the second data source have no `partial_or_full` label. Thus, we should treat the prediction of `partial_or_full` and `cancel_not_for_all_passengers` separately. 

## Failure Dataset

The failure dataset consists of the following two sources:

1. Conversations flowing into `agent` will exit with code `USER_ASKED_FOR_AGENT`. Customers asking for human agent may be due to the fact that Ava's worflow fails to process the customers' requests successfully. 57 sessions (04222025-08182025)

2. Customers decide not to proceed with cancellation at `Confirm cancel` node. The reason may be that Ava's workflow incorrectly brings the customers to this node. 1.8k sessions (04222025-08182025)

For both of the two sources, LLM-as-judge is needed to further identify why Ava's workflow fails to process the customers' requests automatically. The possible reasons are as follows. 

+ the first data source
    + + customers prefer human agents and directly ask for human assistant
    + + the labels predicted by `identify partial` are incorrect and the customers are incorrectly brought to the `confirm leg` node
    + + ...
+ the second data source
    + + the customer does not want to cancel the flight anymore
    + + Ava's workflow incorrectly bring the customer to the `Confirm cancel` node. For instance, the labels predicted by `identify partial` are incorrect. 

# Snowflake Query

## fact_ava_flow
This database can be filtered for desired channel ids
### success

```sql
select *
from dwh.data_warehouse.fact_ava_flow
where IS_FLOW_ABANDONED = False and FLOW = 'cancel_booking' and BOOKING_TYPE = 'flight' and FLOW_START_DATETIME > '2025-04-21' and FLOW_END_DATETIME < '2025-08-19' and OUTCOME = 'success'
```

### cancel_not_for_all_passengers=true

First find all the sessions that dirctly exit after `identify partial` using the following query: 

```sql
select *
from dwh.data_warehouse.fact_ava_flow
where IS_FLOW_ABANDONED = False and FLOW = 'cancel_booking' and BOOKING_TYPE = 'flight' and FLOW_START_DATETIME > '2025-04-21' and FLOW_END_DATETIME < '2025-08-19' AND CAST(GET(EXIT_CODE_ARRAY [0], 'thinking') AS TEXT) ILIKE '%Cant cancel boooking for fewer than all passengers in the booking%'
```

The retrieved data should be further cleaned as some conversation has no following up human agent involved which should be followed by. 
### human_gent

```sql
select *
from dwh.data_warehouse.fact_ava_flow
where IS_FLOW_ABANDONED = False and FLOW = 'cancel_booking' and BOOKING_TYPE = 'flight' and FLOW_START_DATETIME > '2025-04-21' and FLOW_START_DATETIME < '2025-08-19' and OUTCOME = 'agent'
AND CAST(GET(EXIT_CODE_ARRAY [0], 'reason') AS TEXT) ILIKE '%USER_ASKED_FOR_AGENT%'
```

### confirm_cancel_exit

```sql
select *
from dwh.data_warehouse.fact_ava_flow
where IS_FLOW_ABANDONED = False and FLOW = 'cancel_booking' and BOOKING_TYPE = 'flight' and FLOW_START_DATETIME > '2025-04-21' and FLOW_END_DATETIME < '2025-08-19' AND CAST(GET(EXIT_CODE_ARRAY [0], 'thinking') AS TEXT) ILIKE '%User decided not proceed with cancellation%'
```

## ava_chat_history_api_raw_data
This database can be filtered to collect the whole conversations between the customer and the Ava, customer and the human agent if exists.

```sql
select *
from dwh.data_warehouse.ava_chat_history_api_raw_data
where CHANNEL_SID in (
  select DISTINCT CHANNEL_SID
  from the query from the above section
)
```

### success

```sql
select *
from dwh.data_warehouse.ava_chat_history_api_raw_data
where CHANNEL_SID in (
select DISTINCT CHANNEL_SID
from dwh.data_warehouse.fact_ava_flow
where IS_FLOW_ABANDONED = False 
and FLOW = 'cancel_booking' 
and BOOKING_TYPE = 'flight' 
and FLOW_START_DATETIME > '2025-04-22' and FLOW_END_DATETIME < '2025-08-19' 
and OUTCOME = 'success'
) 
and CHANNEL_AVA_TRANSFER_TO_AGENT = FALSE 
and CHANNEL_LAST_AVA_FLOW = 'cancel_booking' 
and IS_AVA_INTERACTION_ACTIONABLE = True
```
some sessions have following up converstaion with human agent. To make sure the flight is canceled with no accident errors, further filters are needed:

```
CHANNEL_AVA_TRANSFER_TO_AGENT = FALSE and CHANNEL_LAST_AVA_FLOW = 'cancel_booking'
```
There are 12 `IS_AVA_INTERACTION_ACTIONABLE=False` and 1927 `IS_AVA_INTERACTION_ACTIONABLE=True`. Need to confirm the definition of `IS_AVA_INTERACTION_ACTIONABLE=False` with Hen.

Every column has only one value. 

Since all the sessions end without the human agent assistant, we can assume all the labels from `identify partial` node are correct. All the sessions have `cancel_not_for_all_passengers=False`, `partial_or_full` may be `FULL` or `PARTIAL`. Further processing is needed to identify `partial_or_full` label. If the session has visited `confirm leg` node, then this session should have label `PARTIAL`, otherwise `FULL`. 

### cancel_not_for_all_passengers=true

```sql
select *
from dwh.data_warehouse.ava_chat_history_api_raw_data
where CHANNEL_SID in (
select DISTINCT CHANNEL_SID
from dwh.data_warehouse.fact_ava_flow
where IS_FLOW_ABANDONED = False 
and FLOW = 'cancel_booking' 
and BOOKING_TYPE = 'flight' 
and FLOW_START_DATETIME > '2025-04-22' and FLOW_END_DATETIME < '2025-08-19' 
AND CAST(GET(EXIT_CODE_ARRAY [0], 'thinking') AS TEXT) ILIKE '%Cant cancel boooking for fewer than all passengers in the booking%'
) 
and CHANNEL_AVA_TRANSFER_TO_AGENT = TRUE 
and CHANNEL_LAST_AVA_FLOW = 'cancel_booking'
and LENGTH(TRIM(AGENT_PART_BODY)) > 0
```
There are 207 sessions. We can assume `cancel_not_for_all_passengers=true` with high possibility. We can also further check the label using LLM-as-judge.

### human_agent

```sql
select *
from dwh.data_warehouse.ava_chat_history_api_raw_data
where CHANNEL_SID in (
select DISTINCT CHANNEL_SID
from dwh.data_warehouse.fact_ava_flow
where IS_FLOW_ABANDONED = False 
and FLOW = 'cancel_booking' 
and BOOKING_TYPE = 'flight' 
and FLOW_START_DATETIME > '2025-04-22' and FLOW_START_DATETIME < '2025-08-19' and OUTCOME = 'agent' 
AND CAST(GET(EXIT_CODE_ARRAY [0], 'reason') AS TEXT) ILIKE '%USER_ASKED_FOR_AGENT%'
)
and LENGTH(TRIM(AGENT_PART_BODY)) > 0;
```
50 sessions. Do not know why the user asks for human agent and LLM-as-judge is needed.

### confirm_cancel_exit

```sql
select *
from dwh.data_warehouse.ava_chat_history_api_raw_data
where CHANNEL_SID in (
select DISTINCT CHANNEL_SID
from dwh.data_warehouse.fact_ava_flow
where IS_FLOW_ABANDONED = False 
and FLOW = 'cancel_booking' 
and BOOKING_TYPE = 'flight' 
and FLOW_START_DATETIME > '2025-04-22' and FLOW_END_DATETIME < '2025-08-19' 
AND CAST(GET(EXIT_CODE_ARRAY [0], 'thinking') AS TEXT) ILIKE '%User decided not proceed with cancellation%'
) 
and CHANNEL_AVA_TRANSFER_TO_AGENT = TRUE 
and LENGTH(TRIM(AGENT_PART_BODY)) > 0 
and CHANNEL_AVA_TRANSFER_TO_AGENT_DOMAIN = 'travel' 
and CHANNEL_AVA_TRANSFER_TO_AGENT_PRODUCT = 'flight'
and CHANNEL_LAST_AVA_FLOW = 'cancel_booking'
```
757 sessions. Some columns' definitions need to be confirmed with Hen. The user may enter flight cancellation workflow more than one time. Need to check the `VISITED_NODES_ARR` to find when `confirm cancel` node was visted and according to the time find the corresponding ip conversation.

```
CHANNEL_AVA_TRANSFER_TO_AGENT_EXPERIENCE:
transactional
undetermined
informational

CHANNEL_AVA_TRANSFER_TO_AGENT_TASK:
```

## IP Conversation
To collect the exact conversations starting from the begining until to the `identify partial` node, run the following script:

```python
python -m src.utils.ip_con_mining_util
```

# Data process

```
(.venv) zpeng@NVNJ0LC222QVJ AgenticWorkflowOptimization % python -m src.utils.filter_partial_util
partial_num: 82
total_num: 1927
done
```

```
ip_con_process_util.py
📊 Visual Distribution of Paired Messages:
============================================================
1 pair(s):  5272 channels ( 94.7%) ██████████████████████████████████████████████████
2 pair(s):   261 channels (  4.7%) ██
3 pair(s):    24 channels (  0.4%) 
4 pair(s):     8 channels (  0.1%) 
5 pair(s):     1 channels (  0.0%) 

🎯 Key Insights:
----------------------------------------
• 5,272 channels (94.7%) have exactly 1 conversation pair
• 294 channels (5.3%) have multiple conversation pairs
• Maximum conversation pairs in a single channel: 5
• Total conversation pairs across all channels: 5,903
```