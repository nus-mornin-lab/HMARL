# HMARL
This Repository contains all the codes used in the paper titled 'Demystifying Complex Treatment Recommendations: A Hierarchical Cooperative Multi-Agent RL Approach' published in IJCNN 2023 conference.
Please cite the paper.

## Prerequisites
- Python version >= 3.5.2
- TensorFlow 1.14.0


## Project structure
- `data` : create a folder named 'data' to save the data prepared from the preprocessing script that is used for training.
- `sql` : data extraction sql codes using MIMIC-IV database. Run the query and save results to data dir with the same name of the queries respectively.
- `preprocessing` : read multiple extracted data tables from sql codes, and save processed data results to data dir
- `main_discrete` : Discrete action space implementation of algorithm, neural networks, `setting.py` containing all hyperparameters.
- `main_continuous` : Continuous action space implementation of algorithm, neural networks, `setting.py` containing all hyperparameters.
- `additional_resources` : contains architecture diagrams and additional experiemental results that could not be included in the paper due to space limitations.
- `baseline` : contains the codes of Qmix baseline for both continuos and discrete action spaces. Most other baselines used the codes shared by the authors.

## Training
Extract data from MIMIC-IV database, perform the preprocessing code `preprocess_4h_mimic.py` and save the processed data to `data` dir
For all algorithms, 
- `cd` into the `HMARL_*` folder, `HMARL_Discrete` is used for training with discrete action space and `HMARL_Continuous` is for continuous action space.
- Execute contextual state scripts using `integrate_previous_steps.py`
- Execute training script, e.g. `python train_*.py  -train_FM 1 -e 1`, if training the state and contextual feature embedding and using them as inputs to the model. Otherwise, set -e and -train_FM to 0.
- All models of Root agent, IV-only agent, Vaso-only agent, and Qmix agent will be saved to `models` dir under `train_*` respectively.
- Training results will be saved to `train_Embedding` or `train` depending on argument -e 1 or -e 0

## Testing
- Testing is automatically executed after training. Testing results will be saved to `test_Embedding` or `test` depending on argument -e 1 or -e 0
