# AI3601RL 
## Bolun Zhang, Nange Wang, Tianhua Li
## Offline Dataset

The offline dataset in `collected_data` consists of offline samples in two tasks (walker_run and walker_walk), each having two levels of data qualities (medium and medium-replay), where the medium data has higher overall quality than medium-replay. The data was collected in the replay buffer when training a TD3 agent in the walker_walk and walker_run environment. The folder `custom_dmc_tasks` contains the environment specifications for the tasks. In the example given in `agent_example.py`, you may refer to the `load_data` function for offline data loading and the `eval` function for testing in these environments.

## Package Specifications

- `dm_control==1.0.14`
- `gym==0.21.0`
- `mujoco==2.3.7`

## Train & Evaluate
### CQL
Use `bash train.sh` on bash, you can change your hyperparameter of the model in this file.

Use `bash evaluate.sh` on bash, you **must** set all hyperparameters of the specific model you want to evaluate manually, details for hyperparameters can be found in the checkpoint dir `configs.txt`.

### CDS
Run `agent_train_eval_CDS.py` directly to train and save the model. 

To change parameters, check **first few lines** of the main function in `agent_train_eval_CDS.py`.
