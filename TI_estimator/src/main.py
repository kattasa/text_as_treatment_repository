import pickle as pkl
from Qmod import *
import numpy as np

from simulation import run_simulation, train_test_split
from argparse import ArgumentParser
import os

def winsorize(x, threshold = 0.05):
    x = np.array(x)
    x[x < threshold] = threshold
    x[x > 1 - threshold] = 1- threshold
    return x

def main():

    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='Path to dataset')
    parser.add_argument('--run_cb', default=False, action='store_true', help='Whether to run causal bert or not.')
    parser.add_argument('--no_simulate', dest='simulate', default=True, action='store_false', help='Whether to simulate outcomes or not')
    parser.add_argument('--task_id', type=int, help='Which row of the file path we should')
    parser.add_argument('--file_path', type=str, default='./file_paths.csv', help='Which dataset contains file paths for the simulations')

    args = parser.parse_args()
    if args.data is None:
        args.data = pd.read_csv(args.file_path).loc[args.task_id, ]['file_name']
    print(args.data)
    df = pd.read_csv(args.data, sep='\t')
    df['text'] = df['text'].map(lambda x: x.lower() if isinstance(x,str) else x)
    df['T'] = df['T_proxy']
    
    np.random.seed(42) # set seed for train/test split
    df, test_df = train_test_split(df)

    if not os.path.exists(f'ih_data_models/{args.data[:-4]}/'):
        os.makedirs(f'ih_data_models/{args.data[:-4]}/')
    mod = QNet(batch_size = 64, # batch size for training
            a_weight = 0.1,  # loss weight for A ~ text -- default parameter but no guidance on selecting optimal hyperparameters
            y_weight = 0.1,  # loss weight for Y ~ A + text
            mlm_weight=1.0,  # loss weight for DistlBert
            modeldir=f'./ih_data_models/{args.data[:-4]}') # directory for saving the best model
            
    mod.train(df['text'].to_numpy(),  # texts in training data
            df['T'].to_numpy(),     # treatments in training data
            df['C'].to_numpy(),     # confounds in training data, binary
            df['Y'].to_numpy(),     # outcomes in training data
            epochs=20,   # the maximum number of training epochs
            learning_rate = 2e-5)  # learning rate for the training

    Q0, Q1, A, Y, _ = mod.get_Q(test_df['text'], test_df['T'], test_df['C'], test_df['Y'])
    Q0 = np.array(Q0)
    Q1 = np.array(Q1)
    A = np.array(A)
    Y = np.array(Y)
    g = get_propensities(A, Q0, Q1, 
                        model_type='RandomForest', # choose the nonparametric model
                        kernel=None,    # kernel function for GPR
                        random_state=0) # random seed for GPR 
    g = np.array(g)

    ih_data = pd.DataFrame({
        'Q0' : Q0,
        'Q1' : Q1,
        'A' : A,
        'Y' : Y,
        'g' : g
    })
    ih_data.to_csv(args.data[:-4] + '_aipw_components.csv', index = False)

    # ti_df = get_TI_estimator(g, Q0, Q1, A, Y, 
                    # error=0.05)  # error bound for confidence interval
    ## use AIPW by setting g not equal to None; otherwise does Q1 - Q0 only
    ## winsorize the propensity scores to avoid extreme values
    ih_data['g'] = winsorize(ih_data.g.values)
    ti_df = ate_aiptw(ih_data.Q0, ih_data.Q1, ih_data.A, ih_data.Y, g=ih_data.g, weight=False, error_bound=0.01)
    pd.DataFrame(ti_df).to_csv(args.data[:-4] + '_tx_effects_point_estimate.csv', index = False)
    ih_data.to_csv(args.data[:-4] + '_aipw_components.csv', index = False)
    print('Prop scores', ih_data.g.max(), ih_data.g.min())
    print('Saved dataframes')

def est_ate_from_aipw_df():
    
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='Path to dataset')
    parser.add_argument('--run_cb', default=False, action='store_true', help='Whether to run causal bert or not.')
    parser.add_argument('--no_simulate', dest='simulate', default=True, action='store_false', help='Whether to simulate outcomes or not')
    parser.add_argument('--task_id', type=int, help='Which row of the file path we should')
    parser.add_argument('--file_path', type=str, default='./file_paths.csv', help='Which dataset contains file paths for the simulations')

    args = parser.parse_args()
    if args.data is None:
        args.data = pd.read_csv(args.file_path).loc[args.task_id, ]['file_name']
    print(args.data)

    ih_data = pd.read_csv(args.data[:-4] + '_aipw_components.csv')
    ih_data['g'] = get_propensities(ih_data.A, ih_data.Q0, ih_data.Q1, model_type='RandomForest')
    
    ih_data['g_win'] = winsorize(ih_data.g.values, threshold = 0.1)
    ti_list = ate_aiptw(ih_data.Q0, ih_data.Q1, ih_data.A, ih_data.Y, g=ih_data.g_win, weight=False, error_bound=0.01)
    ate_winsorized = ti_list[0]
    # ih_data_trimmed = ih_data.loc[(ih_data['g'] >= 0.05) & (ih_data['g'] <= 0.95), ]
    ih_data_trimmed = ih_data.loc[(ih_data['g'] >= 0.1) & (ih_data['g'] <= 0.9), ]
    ti_list = ate_aiptw(ih_data_trimmed.Q0, ih_data_trimmed.Q1, ih_data_trimmed.A, ih_data_trimmed.Y, g=ih_data_trimmed.g_win, weight=False, error_bound=0.01)
    ate_trimmed = ti_list[0]
    ate_or = (ih_data.Q1.values - ih_data.Q0.values).mean()
    ti_df = pd.DataFrame({'ti_aipw_trimmed' : [ate_trimmed], 'ti_aipw_winsorized' : [ate_winsorized], 'ti_or' : [ate_or]})
    ti_df.to_csv(args.data[:-4] + '_tx_effects_point_estimate.csv', index = False)
    ih_data.to_csv(args.data[:-4] + '_aipw_components.csv', index = False)
    print('Prop scores', ih_data.g.max(), ih_data.g.min())
    print('Saved dataframes')
if __name__ == '__main__':
    main()
    est_ate_from_aipw_df()