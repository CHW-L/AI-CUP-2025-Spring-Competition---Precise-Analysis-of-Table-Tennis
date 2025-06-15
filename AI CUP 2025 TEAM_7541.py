# -*- coding: utf-8 -*-
"""
Created on Fri May 30 21:33:16 2025

@author: 林
"""

import re, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.stats import skew, kurtosis
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
dir_train = Path('39_Training_Dataset/39_Training_Dataset')
dir_test  = Path('39_Test_Dataset/39_Test_Dataset')

def binarize(x):
    if isinstance(x, pd.Series):
        return (x == 1).astype(int)
    return int(x == 1)

def safe_skew(x):  return 0.0 if np.std(x) < 1e-8 else skew(x, bias=False)
def safe_kurt(x):  return 0.0 if np.std(x) < 1e-8 else kurtosis(x, bias=False)

fun = {'mean': np.mean, 'std': np.std, 'min': np.min, 'max': np.max,
       'median': np.median, 'q25': lambda x: np.percentile(x, 25),
       'q75': lambda x: np.percentile(x, 75),
       'iqr': lambda x: np.percentile(x, 75)-np.percentile(x, 25),
       'rms': lambda x: np.sqrt(np.mean(x**2)), 'ptp': np.ptp,
       'skew': safe_skew, 'kurt': safe_kurt}   #8軸*12統計

def one_seg(seg: pd.DataFrame) -> pd.Series:
    return pd.Series({f'{c}_{n}': f(seg[c].values.astype(np.float32)) 
                      for c in seg.columns for n, f in fun.items()})

def uid_feature(row: pd.Series, dir_: Path, data_: str, n_seg: int = 27) -> pd.Series:
    uid = row['unique_id']
    raw = pd.read_csv(dir_ / data_ / f'{uid}.txt', sep=' ', header=None,
                      names=['Ax','Ay','Az','Gx','Gy','Gz'])
    
    raw['Acc_mag'] = np.sqrt(raw.Ax**2 + raw.Ay**2 + raw.Az**2)
    raw['Gyr_mag'] = np.sqrt(raw.Gx**2 + raw.Gy**2 + raw.Gz**2) #加2軸
    
    cp = list(map(int, re.findall(r'\d+', row['cut_point'])))
    if not cp or cp[0] != 0:
        cp = [0] + cp
    while len(cp) < n_seg + 1:
        cp.append(len(raw))
    if cp[-1] < len(raw):
        cp.append(len(raw))
    cp = cp[:n_seg+1]
    
    seg_feat = [one_seg(raw.iloc[cp[i]:cp[i+1]]) for i in range(n_seg)]
    seg_df   = pd.DataFrame(seg_feat)
    flat_vals = seg_df.to_numpy().ravel()
    flat_cols = [f's{i}_{col}' for i in range(n_seg) for col in seg_df.columns]
    flatten   = pd.Series(flat_vals, index=flat_cols)
    ms = pd.concat([seg_df.mean().add_prefix('seg_mean_'),
                    seg_df.std().add_prefix('seg_std_')]) #mean/std(16維)
    
    dur = np.diff(cp)
    dur_feat = pd.Series({f'dur_s{i}': dur[i] for i in range(n_seg)} | {
        'dur_mean': dur.mean(), 'dur_std': dur.std(),
        'dur_min': dur.min(), 'dur_max': dur.max()}) #時間節拍(31維)
    
    fft_feat = {} #FFT頻域特徵(6維)
    for col_prefix in ['Acc_mag', 'Gyr_mag']:
        sig = raw[col_prefix].values.astype(np.float32)
        fft_vals = np.fft.rfft(sig)
        power = np.abs(fft_vals)**2
        freq = np.fft.rfftfreq(len(sig), d=1.0)
        if len(power) > 1:
            peak_freq = freq[np.argmax(power)]
            spectral_cent = np.sum(freq*power) / (power.sum()+1e-9)
            low_band_frac = power[freq<=freq.max()*0.1].sum() / (power.sum()+1e-9)
        else:
            peak_freq = spectral_cent = low_band_frac = 0.0
        fft_feat.update({f'{col_prefix}_peakf': peak_freq,
                         f'{col_prefix}_centf': spectral_cent,
                         f'{col_prefix}_lowE': low_band_frac})
    fft_feat = pd.Series(fft_feat)
    
    meta = pd.Series({'mode': row['mode']}) #mode(1維)
    return pd.concat([flatten, ms, dur_feat, fft_feat, meta])

Level_Map          = {2:0, 3:1, 4:2, 5:3}
Level_Map_inverse  = {v:k for k, v in Level_Map.items()}

def make_xy(info_df: pd.DataFrame, target: str, dir_: Path, data_: str):
    X_row, y_row = [], []
    for _, r in tqdm(info_df.iterrows(), total=len(info_df), desc=f'extract {target}'):
        X_row.append(uid_feature(r, dir_, data_))
        if target in ('gender', 'hold racket handed'):
            y_row.append(binarize(r[target]))
        elif target == 'level':
            y_row.append(Level_Map[r[target]])
        else:
            y_row.append(r[target])
    X_df = pd.DataFrame(X_row).astype(np.float32).fillna(0)
    y_sr = pd.Series(y_row, name=target)
    return X_df, y_sr

def rr(df,cols:list[str]) ->None:
    df[cols] = df[cols].round(4) 
    r_sum = df[cols].sum(axis=1)
    mask = r_sum.ne(1.0) 
    if mask.any():
        df.loc[mask,cols] = (df.loc[mask,cols]
                             .div(r_sum[mask],axis=0)).round(4)

    diff = 1.0 - df[cols].sum(axis=1)
    tiny = diff.abs().gt(0)
    df.loc[tiny, cols[-1]] += diff[tiny]

def safe_auc(y_true, y_pred, multi=False):
    uniq = np.unique(y_true)
    if uniq.size < 2:
        return np.nan
    if not multi or uniq.size == 2:
        if y_pred.ndim == 2:
            y_pred = y_pred[:, 1]
        return roc_auc_score(y_true, y_pred)
    return roc_auc_score(y_true, y_pred, average='micro', multi_class='ovr')

def lgb_cv(X, y, groups, lgb_params, num_class=1, n_fold=5):
    sgkf = StratifiedGroupKFold(n_splits=n_fold, shuffle=True, random_state=42)
    models, aucs = [], []
    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups)):
        dtr = lgb.Dataset(X.iloc[tr], y.iloc[tr])
        dva = lgb.Dataset(X.iloc[va], y.iloc[va])
        m = lgb.train(lgb_params, dtr, 4000,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False),
                                 lgb.log_evaluation(0)])
        models.append(m)
        
        pred = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        auc = safe_auc(y.iloc[va], pred, multi=(num_class > 2))
        aucs.append(auc)
        print(f'Fold {fold} AUC={auc:.4f}')
    print('CV AUC=', np.nanmean(aucs))
    return models

def ens_predict(models, X, num_class=1):
    return np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)

PARAM_BIN = dict(
    objective='binary', metric='auc',
    learning_rate=0.03, num_leaves=96,
    min_data_in_leaf=30, feature_fraction=0.6,
    bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, min_gain_to_split=0.01,
    seed=42, verbosity=-1)

def param_multi(nc):
    p = PARAM_BIN.copy()
    p.update(objective='multiclass', metric='multi_logloss', num_class=nc)
    return p

train_info = pd.read_csv(dir_train/'train_info.csv')
test_info  = pd.read_csv(dir_test/'test_info.csv')

test_feat = [uid_feature(r, dir_test, 'test_data') 
             for _, r in tqdm(test_info.iterrows(), total=len(test_info), desc='extract test')]
test_X = pd.DataFrame(test_feat).astype(np.float32).fillna(0)

submission = pd.DataFrame({'unique_id': test_info['unique_id']})
task_cfg = {'gender': (PARAM_BIN, 1),
            'hold racket handed': (PARAM_BIN, 1),
            'play years': (param_multi(3), 3),
            'level': (param_multi(4), 4)}

for tgt, (params, n_cls) in task_cfg.items():
    print(f'\n=== Training {tgt} ===')
    X, y = make_xy(train_info, tgt, dir_train, 'train_data')
    
    grp = train_info['player_id']
    need_f = 5
    grp_cnt = (pd.concat([grp, y], axis=1)
               .groupby(y.name)['player_id'].nunique())
    n_fold_use = max(2, min(need_f, grp_cnt.min()))
    if n_fold_use != need_f:
        print(f'類別不足，下修為{n_fold_use} folds')
    
    models = lgb_cv(X, y, grp, params, num_class=n_cls, n_fold=n_fold_use)
    pred = ens_predict(models, test_X, num_class=n_cls)
    
    if n_cls == 1:
        submission[tgt] = pred.round(4)
    elif tgt == 'play years': 
        for i in range(3):
            submission[f'play years_{i}'] = pred[:, i]
        rr(submission, [f'play years_{i}' for i in range(3)])
    else:
        for i in range(pred.shape[1]):
            real_lv = Level_Map_inverse[i]   # 2/3/4/5
            submission[f'level_{real_lv}'] = pred[:, i]
        rr(submission, [f'level_{k}' for k in (2, 3, 4, 5)])

submission = submission.sort_values('unique_id')
submission.to_csv('submission_f.csv', index=False)