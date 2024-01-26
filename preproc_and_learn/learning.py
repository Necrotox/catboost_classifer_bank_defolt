import pandas as pd
import dill
from catboost import Pool, CatBoostClassifier, CatBoostRanker
import time
import warnings


df = pd.read_parquet('/data/train_target.pq')

features = [i for i in df.columns if i not in ['id', 'flag']]
features_for_split = [i for i in df.columns if i not in ['id']]
target = 'flag'
model_catboost = CatBoostClassifier(
    verbose=3000,
    loss_function='Logloss',
    eval_metric='AUC',
    task_type="GPU",
    auto_class_weights='Balanced',
    depth=5,
    l2_leaf_reg=13,
    random_state=42,
    iterations=100000,
    learning_rate=0.001
)

model_catboost.fit(
    df[features], df[target],
    eval_set=(df[features], df[target]),
    plot=True
)

with open(f'G:/model/catboost_model_{datetime.datetime.now().strftime("%Y%m%d%H%M")}.pkl', "wb") as file:
    dill.dump({'model': model_catboost,
               'metadata': {
                   'name': 'Catboostclassifer bank default',
                   'author': 'Viktor Aleksandrov',
                   'version': 1,
                   'module': datetime.datetime.now(),
                   'model_name': type(model_catboost).__name__,
               }}, file)