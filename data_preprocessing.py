import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm


def add_confounding_outcomes(qlevel_results_topic, outcomes, conf_col):
    '''
    Add confounding to the outcomes
    ---
    qlevel_results_topic: pd.DataFrame, the data frame with outcomes
    outcomes: list, the list of outcomes
    conf_col: str, the name of the column that contains the confounding variable
    ---
    Return: list of str, the names of new columns based on level of confounding; pd.DataFrame, the data frame with confounded outcomes
    '''

    # for each outcome
    for outcome in outcomes:
        # rename the original outcome to outcome + 'Conf_original'
        qlevel_results_topic[outcome + 'Conf_original'] = qlevel_results_topic[outcome].values
        # add 1 to all treated texts for treatment effect without confounding but still a non-negligible effect
        qlevel_results_topic[outcome + 'Conf_no'] = qlevel_results_topic[outcome] + (qlevel_results_topic['ih_final'] == 'y').astype(int)
        # add 1 to all gc texts for non-text-level confounding
        qlevel_results_topic[outcome + 'Conf_gc'] = qlevel_results_topic[outcome] + ((qlevel_results_topic['review_topic'] == 'gc')).astype(int)
        # add 1 to all gc texts for treatment effect
        qlevel_results_topic[outcome + 'Conf_gc'] = qlevel_results_topic[outcome + 'Conf_gc'] + (qlevel_results_topic['ih_final'] == 'y').astype(int)
        # add 2 (subtract 2) to all posConf (not posConf) texts outcomes for text-level confounding
        qlevel_results_topic[outcome + 'Conf_conf'] = qlevel_results_topic[outcome] + 2 * (2 * (qlevel_results_topic[conf_col] >= 50) - 1).astype(int)
        # add 2 (subtract 2) to all treated (control) obs' outcomes for text-level confounding
        qlevel_results_topic[outcome + 'Conf_conf'] = qlevel_results_topic[outcome + 'Conf_conf'] + 2 * (qlevel_results_topic['ih_final'] == 'y').astype(int)
    # save only the relevant data
    # outcomes_expanded = [x + 'Conf_original' for x in outcomes] + [x + 'Conf_no' for x in outcomes] + [x + 'Conf_gc' for x in outcomes] + [x + 'Conf_conf' for x in outcomes]
    outcomes_expanded = [x + 'Conf_original' for x in outcomes] + [x + 'Conf_conf' for x in outcomes]
    qlevel_results = qlevel_results_topic[['ResponseId', 'ih_final', 'qid', 'text', 'review_topic', 'w3_qid', conf_col]]
    # binarize the outcomes
    qlevel_results = pd.concat([qlevel_results, (qlevel_results_topic[outcomes_expanded] >= 3).astype(int)], axis = 1)

    return outcomes_expanded, qlevel_results

    
def add_confounding_treatment(qlevel_results, conf_col, seed):
    '''
    sample observations to simulate selection bias
    ---
    qlevel_results: pd.DataFrame, the data frame with confounded outcomes
    conf_col: str, the name of the column that contains the confounding variable
    seed : int, the seed for random sampling
    ---
    Return: pd.DataFrame, the data frame with confounded treatment
    '''
    # label the observation as original text or edited
    qlevel_results['original_text'] = (qlevel_results.w3_qid.str.split('_', expand = True)[3] == '0')
    # label the observation as posConf or not
    qlevel_results['posConf'] = qlevel_results[conf_col] >= 50

    np.random.seed(seed)
    # for each QID (original-text-level group)
        # if the original post was posConf and IH, sample 2 originals and 1 edit
        # if the original post was not posConf and IH, sample 1 originals and 2 edits
        # if the original post was posConf and not IH, sample 1 originals and 2 edits
        # if the original post was not posConf and not IH, sample 2 original and 1 edits

    new_df = []
    for qid in qlevel_results.qid.unique():
        original_posConf_ih = qlevel_results.query(f'qid == "{qid}"').query('original_text == True').query('posConf == True').query('ih_final == "y"')
        original_not_posConf_ih = qlevel_results.query(f'qid == "{qid}"').query('original_text == True').query('posConf == False').query('ih_final == "y"')
        original_posConf_not_ih = qlevel_results.query(f'qid == "{qid}"').query('original_text == True').query('posConf == True').query('ih_final == "n"')
        original_not_posConf_not_ih = qlevel_results.query(f'qid == "{qid}"').query('original_text == True').query('posConf == False').query('ih_final == "n"')
        
        if not original_posConf_ih.empty:
            new_df.append(original_posConf_ih.sample(n=2, replace=False))
            new_df.append(qlevel_results.query(f'qid == "{qid}"').query('original_text == False').sample(n=1, replace=False))
        elif not original_not_posConf_ih.empty:
            new_df.append(original_not_posConf_ih.sample(n=1, replace=False))
            new_df.append(qlevel_results.query(f'qid == "{qid}"').query('original_text == False').sample(n=2, replace=False))
        elif not original_posConf_not_ih.empty:
            new_df.append(original_posConf_not_ih.sample(n=1, replace=False))
            new_df.append(qlevel_results.query(f'qid == "{qid}"').query('original_text == False').sample(n=2, replace=False))
        elif not original_not_posConf_not_ih.empty:
            new_df.append(original_not_posConf_not_ih.sample(n=2, replace=False))
            new_df.append(qlevel_results.query(f'qid == "{qid}"').query('original_text == False').sample(n=1, replace=False))
    new_df = pd.concat(new_df).reset_index(drop=True)
    return new_df


def export_w_outcome(new_df, outcome, directory, file_prefix = 'ih_data'):
    '''
    save the data with a specific outcome as a tsv file for baselines
    ---
    new_df: pd.DataFrame, the data frame with data
    outcome: str, the name of the outcome to be exported
    directory: str, the name of the directory where the data will be stored
    file_prefix: str, what should the prefix of the file name be?
    ---
    Return: name of the file path
    '''
    causal_text_df = pd.DataFrame({
        'text' : new_df.text,
        'Y' : (new_df[outcome]).astype(int),
        'C' : (new_df.review_topic == 'im').astype(int), # one categorical covariate encoded as a binary vector
        'T_proxy' : (new_df.ih_final == 'y').astype(int)
        })

    

    output_file = f'{directory}/{file_prefix}_{outcome}.tsv'
    output_file_path = Path(output_file)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    causal_text_df.to_csv(output_file, index = False, sep = '\t')

    return f'./{file_prefix}_{outcome}.tsv'
    


def main():
    conf_col = 'respect'

    ## read data: outcomes for each observations
    qlevel_results_raw = pd.read_csv('./qlevel_final.csv')
    qlevel_results_topic = qlevel_results_raw.query('review_topic != "cc" ')

    ## ratings on each text
    ratings = pd.read_csv('./rating_data.csv')
    # get original-text-group level ID
    ratings['qid'] = ratings.statement_id.apply(lambda x: x[:-2])
    ratings_gb = ratings.groupby('qid')[[conf_col]].mean().reset_index()

    ## merge ratings to qlevel_results
    qlevel_results_topic = qlevel_results_topic.merge(ratings_gb, on = 'qid', how = 'left')
    
    outcomes = ["enjoy_num","persuasive_num","articulated","informative","aggressive","persuasive_me"]
    
    ## add confounding to outcome
    outcomes_expanded, qlevel_results = add_confounding_outcomes(qlevel_results_topic, outcomes, conf_col)
    ## get only complete case data
    qlevel_results = qlevel_results.dropna()

    ## for a bunch of different iterations, sample different types of data and save them to their respective directories
    output_file_dict_textcause = {'iter' : [], 'file_name' : []}
    output_file_dict_tiestimator = {'iter' : [], 'file_name' : []}

    for iter in tqdm(range(100)):
        seed = 42 * iter + 2024
        new_df = add_confounding_treatment(qlevel_results, conf_col, seed)

        ## save the data as a single dataframe
        Path('./qlevel_results_confounded').mkdir(parents=True, exist_ok=True)
        new_df.to_csv(f'./qlevel_results_confounded/seed{seed}.csv', index = False)
        
        ## save the data for each outcome in the confounded dataset
        for outcome in outcomes_expanded:
            output_file = export_w_outcome(new_df, outcome, directory = './causal-text/src', file_prefix = f'sim_data/seed{seed}/outcome')
            output_file_dict_textcause['iter'].append( iter )
            output_file_dict_textcause['file_name'].append( output_file )

            output_file = export_w_outcome(new_df, outcome, directory = './TI_estimator/src', file_prefix = f'sim_data/seed{seed}/outcome')
            output_file_dict_tiestimator['iter'].append( iter )
            output_file_dict_tiestimator['file_name'].append( output_file )


    pd.DataFrame(output_file_dict_tiestimator).to_csv('./TI_estimator/src/file_paths.csv', index = False)
    pd.DataFrame(output_file_dict_textcause).to_csv('./causal-text/src/file_paths.csv', index = False)

    bash_slurm_script = f'''#!/bin/bash
#SBATCH --job-name=conf
#SBATCH --output=slurm_outputs/array_job_%A_%a.out  # %A is the job array ID, %a is the task ID
#SBATCH --error=slurm_outputs/array_job_%A_%a.err
#SBATCH --array=0-{len(output_file_dict_tiestimator['file_name']) - 1} # Job array range
#SBATCH --ntasks=1                    # Number of tasks per job
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=50G                      # Memory per task
#SBATCH --time=01:00:00               # Time limit
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --exclude=linux[46],linux[1]

srun -u python3 main.py --task_id $SLURM_ARRAY_TASK_ID --no_simulate --run_cb
'''
    ti_estimator_bash_file = Path('./TI_estimator/src/causal_text_slurm.sh')
    with open(ti_estimator_bash_file, 'w') as f:
        f.write(bash_slurm_script)

    causal_text_bash_file = Path('./causal-text/src/causal_text_slurm.sh')
    with open(causal_text_bash_file, 'w') as f:
        f.write(bash_slurm_script)

if __name__ == '__main__':
    main()


