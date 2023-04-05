import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor as RFR


data = pd.read_csv(r'D:\Mashine_Learning\data_lern\insurance.csv', sep=',')
data_train = data.sample(frac=0.9)
data_validate = data.loc[~data.index.isin(data_train.index)]

col_trans_tree_ordinal = ColumnTransformer([
    ('encode_ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['sex', 'smoker', 'region']),
], remainder='passthrough')

pipe_ranfor_ord_encode = Pipeline([('col_trans_tree', col_trans_tree_ordinal), ('ranfor', RFR(n_jobs=6))])

ranfor_params = {
    'ranfor__n_estimators': range(10, 122, 2),
    'ranfor__max_depth': range(3, 10, 1),
    'ranfor__min_samples_leaf': range(5, 50, 1),
    'ranfor__max_features': [2, 3, 4, 5]
}

grid_search_ranfor = GridSearchCV(
    pipe_ranfor_ord_encode,
    ranfor_params,
    scoring='neg_mean_absolute_error',
    n_jobs=1,
    verbose=1,
)


grid_search_ranfor.fit(data_train.loc[:, ~data_train.columns.isin(['charges'])], data_train['charges'])
