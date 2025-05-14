"""

"""

from tqdm import tqdm
import sys
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import random

import torch

from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier, LassoCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from sklearn import feature_extraction

import simulation
import label_expansion
import util

import pickle as pkl
import time

import CausalBert

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from econml.dml import LinearDML
from econml.metalearners import TLearner
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split


def prepare_covariates(df, 
    stopwords=None,
    vocab_size=2000,
    use_counts=False):

    def admissable(w):
        if stopwords is None:
            return True
        return w not in stopwords

    # 2k most common not in lex
    c = Counter([w for s in df['text'] for w in util.word_tokenize(s.lower()) if admissable(w)])
    vocab = list(zip(*c.most_common(vocab_size)))[0]

    # vectorize inputs
    vectorizer = feature_extraction.text.CountVectorizer(
        lowercase=True,
        tokenizer=util.word_tokenize,
        vocabulary=vocab,
        binary=(not use_counts),
        ngram_range=(1, 1))
    corpus = list(df['text'])
    vectorizer.fit(corpus)
    X = np.asarray(vectorizer.transform(corpus).todense())
    return X, vocab, vectorizer


def run_parameterized_estimators(
    df, df2=None,
    stopwords=None,
    vocab_size=2000,
    use_counts=False,
    threshold=0.8,
    only_zeros=True,
    inner_alpha='optimal',
    outer_alpha='optimal',
    g_weight=1, Q_weight=1, mlm_weight=1, run_cb=False):
    """ Run all the ATE estimators based on models:
            regression expansion (+pu classifier), bert adjustment, and
                regression expansion + bert.
    """
    X, vocab, vectorizer = prepare_covariates(df, stopwords, vocab_size, use_counts)
    T_true = df['T_true'].to_numpy()
    T_proxy = df['T_proxy'].to_numpy()

    # PU classifier expansion
    only_zeros=True
    pu = label_expansion.PUClassifier(
        inner_alpha=inner_alpha,
        outer_alpha=outer_alpha)
    pu.fit(X, T_proxy)
    T_plus_pu = label_expansion.expand_variable(pu, X, T_proxy,
        threshold=threshold,
        only_zeros=only_zeros)
    ATE_pu = util.ATE_adjusted(df.C_true, T_plus_pu , df.Y_sim)

    # Plain regression expansion
    reg = SGDClassifier(loss="log_loss", penalty="l2", alpha=outer_alpha)
    reg.fit(X, T_proxy)
    T_plus_reg = label_expansion.expand_variable(reg, X, T_proxy,
        threshold=threshold,
        only_zeros=only_zeros)
    ATE_reg = util.ATE_adjusted(df.C_true, T_plus_reg , df.Y_sim)

    if run_cb:
        cbw = CausalBert.CausalBertWrapper(g_weight=g_weight, Q_weight=args.Q_weight, mlm_weight=args.mlm_weight)
        cbw.train(df['text'], df.C_true, df.T_proxy, df.Y_sim, epochs=3)
        ATE_cb_Tproxy = cbw.ATE(df.C_true, df['text'], Y=df.Y_sim, platt_scaling=False)

        cbw = CausalBert.CausalBertWrapper(g_weight=g_weight, Q_weight=args.Q_weight, mlm_weight=args.mlm_weight)
        cbw.train(df['text'], df.C_true, T_plus_pu, df.Y_sim, epochs=3)
        ATE_cb_Tplus = cbw.ATE(df.C_true, df['text'], Y=df.Y_sim, platt_scaling=False)
        # save cbw
        # with open('./src_outputs/cbw.pkl', 'wb') as f:
        #     pkl.dump(cbw, f)
    else:
        ATE_cb_Tproxy, ATE_cb_Tplus = -1, -1

    return ATE_pu, ATE_reg, ATE_cb_Tproxy, ATE_cb_Tplus


def get_data(args):
    """ Read in a dataset and make sure it has fields
            text, T_true, T_proxy, C_true, Y_sim
    """
    if args.simulate:
        # Add columns T_true T_proxy C_true Y_sim to the data
        df = pd.read_csv(args.data, sep='\t')
        df['text'] = df['text'].map(lambda x: x.lower() if isinstance(x,str) else x)
        df = simulation.run_simulation(df,
            propensities=[args.p1, args.p2] if args.p1 > 0 else None,
            precision=args.pre,
            recall=args.rec,
            b0=args.b0,
            b1=args.b1,
            offset=args.off,
            gamma=args.gamma,
            accuracy=args.acc,
            proxy_type=args.ptype,
            size=args.size)

        # df2 = df[['text', 'Y_sim', 'C_true', 'T_proxy']]
        # df2.to_csv('music_complete.tsv', sep='\t'); quit()
        return [df]
    else:
        # use what's given without any changes
        # (T_true, T_proxy, C_true, and Y should already be in there)
        df = pd.read_csv(args.data, sep='\t')
        df['text'] = df['text'].map(lambda x: x.lower() if isinstance(x,str) else x)
        df['Y_sim'] = df['Y']
        df['C_true'] = df['C']
        
        return [df]


def run_label_expansion(df, args, stopwords=None, use_counts=False, single_class=False, only_zeros=True,
        inner_alpha='optimal', outer_alpha='optimal', threshold=0.8):
    X, vocab, vectorizer = prepare_covariates(df, stopwords, args.vs, use_counts)
    T_proxy = df['T_proxy'].to_numpy()

    #  one-class learning
    if single_class:
        model = label_expansion.PUClassifier(
            inner_alpha=inner_alpha,
            outer_alpha=outer_alpha)

    # use a logistic regression
    else:
        model = SGDClassifier(loss="log_loss", penalty="l2", alpha=outer_alpha)

    model.fit(X, T_proxy)
    T_plus = label_expansion.expand_variable(model, X, T_proxy,
        threshold=threshold,
        only_zeros=only_zeros)
    ATE_plus = util.ATE_adjusted(df.C_true, T_plus , df.Y_sim)

    return ATE_plus, T_plus

######
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LassoCV, LogisticRegressionCV, RidgeCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from econml.dml import LinearDML
from econml.metalearners import TLearner
from scipy.sparse import hstack, csr_matrix
import numpy as np

def bow_aipw(text, Y, T, C, seed = 2024):
    '''
    Estimates the average treatment effect using AIPW estimator on bag-of-words text data.
    Parameters:
    X_text (pd.Series): Series containing text data.
    Y (pd.Series): Series containing the outcome variable.
    T (pd.Series): Series containing the treatment variable.
    C (pd.Series): Series containing the covariate variable.
    Returns:
    ate : float : Average treatment effect.
    '''
    np.random.seed(seed)
    # Step 1: Convert text to bag of words
    vectorizer = CountVectorizer()
    X_text = vectorizer.fit_transform(text)#text.map(lambda x: x.lower() if isinstance(x,str) else x))

    # Step 2: Prepare other variables
    Y = Y.values
    T = T.values
    C = C.values

    # Convert the pandas Series C to a sparse matrix
    C_sparse = csr_matrix(C.reshape(-1, 1))

    # Add the C column to the X_text matrix
    X = hstack([X_text, C_sparse]).toarray()

    # Step 3: Estimate the average treatment effect using RF as outcome regressor and propensity score estimator
    est = LinearDML(model_y=RandomForestRegressor(), model_t=RandomForestClassifier(), discrete_treatment=True)
    est.fit(Y, T, X=X)

    # Step 4: Get the average treatment effect
    ate = est.ate(X=X)

    return ate


def bow_tlearner(text, Y, T, C, seed = 2024):
    '''
    Estimates the average treatment effect using LASSO outcome regression on bag-of-words text data.
    Parameters:
    X_text (pd.Series): Series containing text data.
    Y (pd.Series): Series containing the outcome variable.
    T (pd.Series): Series containing the treatment variable.
    C (pd.Series): Series containing the covariate variable.
    Returns:
    ate : float : Average treatment effect.
    '''
    np.random.seed(seed)
    # Step 1: Convert text to bag of words
    vectorizer = CountVectorizer()
    X_text = vectorizer.fit_transform(text.map(lambda x: x.lower() if isinstance(x,str) else x))

    # Step 2: Prepare other variables
    Y = Y.values
    T = T.values
    C = C.values

    # Convert the pandas Series C to a sparse matrix
    C_sparse = csr_matrix(C.reshape(-1, 1))

    # Add the C column to the X_text matrix
    X = hstack([X_text, C_sparse]).toarray()

    # Step 3: Estimate the average treatment effect using LASSO as an outcome regression
    est = TLearner(models=RandomForestRegressor())
    est.fit(Y, T, X=X)

    # Step 4: Get the average treatment effect
    ate = est.ate(X=X)

    return ate


def bow_ipw(text, Y, T, C, seed = 2024):
    '''
    Estimates the average treatment effect using IPW on bag-of-words text data.
    Parameters:
    X_text (pd.Series): Series containing text data.
    Y (pd.Series): Series containing the outcome variable.
    T (pd.Series): Series containing the treatment variable.
    C (pd.Series): Series containing the covariate variable.
    Returns:
    ate : float : Average treatment effect.
    '''
    def ipw(X, T, Y, learner):
        ## train test split on X, T, Y
        X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.5, random_state=seed)
        learner.fit(X_train, T_train)
        ps = learner.predict_proba(X_test)[:, 1]
        weight = T_test * 1/(ps) - (1-T_test) * 1/(1-ps)
        ate = (Y_test * weight).mean()
        return ate, ps


    np.random.seed(seed)
    # Step 1: Convert text to bag of words
    vectorizer = CountVectorizer()
    X_text = vectorizer.fit_transform(text.map(lambda x: x.lower() if isinstance(x,str) else x))

    # Step 2: Prepare other variables
    Y = Y.values
    T = T.values
    C = C.values

    # Convert the pandas Series C to a sparse matrix
    C_sparse = csr_matrix(C.reshape(-1, 1))

    # Add the C column to the X_text matrix
    X = pd.DataFrame(hstack([X_text, C_sparse]).toarray())

    # Step 3: Estimate the average treatment effect using RF as PS estimator
    return ipw(X, T, Y, RandomForestClassifier())

########

def run_experiment(args):
    """ Run an experiment with the given args and seed.

        Returns {causal estimator: ATE estimate}
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed) 

    df_list = get_data(args)
    print('#DFs considered:', len(df_list))
    
    ATE_estimates = {'ptype' : [args.ptype] * len(df_list),
                    'unadj_T' : [np.nan] * len(df_list),
                    'ate_T' : [np.nan] * len(df_list),
                    'ate_matrix' : [np.nan] * len(df_list),
                    'unadj_T_proxy' : [np.nan] * len(df_list),
                    'ate_T_proxy' : [np.nan] * len(df_list),
                    'ate_T_plus_reg' : [np.nan] * len(df_list),
                    'ate_T_plus_pu' : [np.nan] * len(df_list),
                    'ate_cb_T_proxy' : [np.nan] * len(df_list),
                    'ate_cb_T_plus_pu' : [np.nan] * len(df_list),
                    'ate_cb_T_plus_reg' : [np.nan] * len(df_list),
                    'ate_bow' : [np.nan] * len(df_list),
                    'ate_aipw' : [np.nan] * len(df_list),
                    'ate_ipw' : [np.nan] * len(df_list),
                    'iter' : [np.nan] * len(df_list)}
    for i, df in enumerate(df_list):
        ATE_estimates['iter'] = i
        if 'T_true' in df:
            ATE_estimates['unadj_T'][i] = util.ATE_unadjusted(df.T_true, df.Y_sim)
            ATE_estimates['ate_T'][i] = util.ATE_adjusted(df.C_true, df.T_true, df.Y_sim)
            
            ATE_estimates['ate_matrix'][i] =util.ATE_matrix(df.T_true, df.T_proxy, df.C_true, df.Y_sim)

        if 'T_proxy' in df:
            # estimate ATE using different types of estimators
            ATE_estimates['unadj_T_proxy'][i] =util.ATE_unadjusted(df.T_proxy, df.Y_sim)
            ATE_estimates['ate_T_proxy'][i] =util.ATE_adjusted(df.C_true, df.T_proxy, df.Y_sim)

            ATE_T_plus_reg, T_plus_reg = run_label_expansion(df, args, 
                inner_alpha=args.ina, outer_alpha=args.outa, threshold=args.thre)
            ATE_estimates['ate_T_plus_reg'][i] = ATE_T_plus_reg

            ATE_T_plus_pu, T_plus_pu = run_label_expansion(df, args, single_class=True, 
                inner_alpha=args.ina, outer_alpha=args.outa, threshold=args.thre)
            ATE_estimates['ate_T_plus_pu'][i] = ATE_T_plus_pu

            ATE_estimates['ate_bow'][i] = bow_tlearner(df['text'], df['Y_sim'], df['T_proxy'], df['C_true'])
            ATE_estimates['ate_aipw'][i] = bow_aipw(df['text'], df['Y_sim'], df['T_proxy'], df['C_true'])
            ATE_estimates['ate_ipw'][i], ps_bow = bow_ipw(df['text'], df['Y_sim'], df['T_proxy'], df['C_true'])

            # save results
            pd.DataFrame({'ps' : ps_bow}).to_csv(args.data[:-4] + '_ps_bow.csv', index = False)
            if args.run_cb:
                df, df_test = train_test_split(df, test_size=0.5, random_state=args.seed)
                cbw = CausalBert.CausalBertWrapper(g_weight=args.g_weight, Q_weight=args.Q_weight, mlm_weight=args.mlm_weight)
                cbw.train(df['text'], df.C_true, df.T_proxy, df.Y_sim, epochs=3)
                ATE_cb_Tproxy, Q1, Q0 = cbw.ATE(df_test.C_true, df_test['text'], Y=df_test.Y_sim, platt_scaling=False)
                ATE_estimates['ate_cb_T_proxy'][i] = ATE_cb_Tproxy
                cb_Thats = cbw.get_tx_preds(texts = df_test['text'], confounds = df_test.C_true, outcome=df_test.Y_sim)
                Q1 = np.array(Q1)
                Q0 = np.array(Q0)
                df_test['Q1'] = Q1
                df_test['Q0'] = Q0
                df_test['That'] = cb_Thats
                df_test.to_csv(args.data[:-4] + '_cbw_components.csv', index = False)
                # save cbw
                if not os.path.exists(f'./src_outputs/{args.data[:-4]}/'):
                    os.makedirs(f'./src_outputs/{args.data[:-4]}/')

                with open(f'./src_outputs/{args.data[:-4]}/cbw_Tproxy.pkl', 'wb') as f:
                    pkl.dump(cbw, f)

                # cbw = CausalBert.CausalBertWrapper(g_weight=args.g_weight, Q_weight=args.Q_weight, mlm_weight=args.mlm_weight)
                # cbw.train(df['text'], df.C_true, T_plus_pu, df.Y_sim, epochs=3)
                # ATE_cb_Tplus = cbw.ATE(df.C_true, df['text'], Y=df.Y_sim, platt_scaling=False)
                # ATE_estimates['ate_cb_T_plus_pu'][i] = ATE_cb_Tplus
                # with open(f'./src_outputs/{args.data[:-4]}/cbw_Tplus.pkl', 'wb') as f:
                #     pkl.dump(cbw, f)

                # cbw = CausalBert.CausalBertWrapper(g_weight=args.g_weight, Q_weight=args.Q_weight, mlm_weight=args.mlm_weight)
                # cbw.train(df['text'], df.C_true, T_plus_reg, df.Y_sim, epochs=3)
                # ATE_cb_Tplus = cbw.ATE(df.C_true, df['text'], Y=df.Y_sim, platt_scaling=False)
                # ATE_estimates['ate_cb_T_plus_reg'][i] = ATE_cb_Tplus
                # with open(f'./src_outputs/{args.data[:-4]}/cbw_Tplus_reg.pkl', 'wb') as f:
                #     pkl.dump(cbw, f)

                ATE_estimates['ate_cb_T_plus_pu'][i] = np.nan
                ATE_estimates['ate_cb_T_plus_reg'][i] = np.nan


    return pd.DataFrame(ATE_estimates)


if __name__ == '__main__':
    from argparse import ArgumentParser
    import os


    parser = ArgumentParser()
    parser.add_argument('--p1', type=float, default=-1, help='P(T* = 0 | C) in simulation (-1 to ignore)')  #0.88
    parser.add_argument('--p2', type=float, default=-1, help='P(T* = 1 | C) in simulation (-1 to ignore)')  #0.842
    parser.add_argument('--ptype', type=str, default='lex', help='type of T*', choices=['random', 'lex'])
    parser.add_argument('--acc', type=float, default=-1, help='T*/T accuracy (-1 to ignore).')
    parser.add_argument('--pre', type=float, default=-1, help='Precision between T* and T (-1 to ignore)')  # 0.94
    parser.add_argument('--rec', type=float, default=-1, help='Recall between T* and T (-1 to ignore)') # 0.98
    parser.add_argument('--b0', type=float, default=0.8, help='Simulated treatment strength') # 0.4, 0.8
    parser.add_argument('--b1', type=float, default=4.0, help='Simulated confound strength')  # -0.4, 4.0
    parser.add_argument('--gamma', type=float, default=1.0, help='Noise level in simulation')  # 0, 1
    parser.add_argument('--off', type=float, default=0.9, help='Simulated offset for T/C pre-threshold means')
    parser.add_argument('--size', type=str, default=-1, help='Sample size if you want to sub-sample the data (-1 to ignore)')
    parser.add_argument('--vs', type=int, default=2000, help='Vocab size for T+ model')
    parser.add_argument('--ina', type=float, default=0.00359, help='One-class regression inner alpha (regularization strength)')
    parser.add_argument('--outa', type=float, default=0.00077, help='One-class regression outer alpha (regularization strength)')
    parser.add_argument('--thre', type=float, default=0.22, help='T+ classifier threshold')
    parser.add_argument('--g_weight', type=float, default=0.0, help='Loss weight for the g head in Causal Bert.')
    parser.add_argument('--Q_weight', type=float, default=0.1, help='Loss weight for the Q head in Causal Bert.')
    parser.add_argument('--mlm_weight', type=float, default=1.0, help='Loss weight for the mlm head in Causal Bert.')
    parser.add_argument('--data', type=str, default=None, help='Path to dataset')
    parser.add_argument('--task_id', type=int, help='Which row of the file path we should')
    parser.add_argument('--file_path', type=str, default='./file_paths.csv', help='Which dataset contains file paths for the simulations')
    parser.add_argument('--seed', type=str, default='420', help='seeds to run')
    parser.add_argument('--run_cb', default=False, action='store_true', help='Whether to run causal bert or not.')
    parser.add_argument('--no_simulate', dest='simulate', default=True, action='store_false', help='Whether to simulate outcomes or not')
    parser.add_argument('--n_bootstraps', dest='n_bootstraps', default=1, action='store_false', help='Whether to simulate outcomes or not. DEPRICATED.')


    args = parser.parse_args()

    # df_list = get_data(args)

    if ',' in args.seed:
        seeds = args.seed.split(',')
    else:
        seeds = [args.seed]

    # args.data will be None in the LLM is too big paper replication because we only pass in the file_path
    if args.data is None:
        ## which dataset do we want to analyze?
        file_path_df = pd.read_csv(args.file_path)
        args.data = file_path_df.loc[args.task_id, ]['file_name']
    print(args.data)
    results = []
    for seed in seeds:
        args.seed = int(seed)

        # Hack to run random + lex
        args.ptype = 'random'
        args.run_cb = False
        result_random = run_experiment(args)
        result_random['ate_T_proxy_random'] = result_random['ate_T_proxy'].values
        results.append(result_random.assign(seed = seed))

        args.ptype = 'lex'
        args.run_cb = True
        result = run_experiment(args)
        results.append(result.assign(seed = seed))
  
    pd.concat(results).to_csv(args.data[:-4] + '_tx_effects_point_estimate.csv', index = False)
    print('Saved dataframe')
    quit()
