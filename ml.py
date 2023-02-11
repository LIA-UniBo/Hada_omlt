import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
import os

HW = ['pc', 'vm', 'g100']
ALG_PARAMS = {
        "convolution" : 4, 
        "saxpy" : 3,
        #"blackscholes" : 15, 
        "correlation" : 7, 
        "fwt" : 2 
        }
TARGET = ["memory", "time"]

def create_dataset_quality(dir_path, algorithm):

    
    data = pickle.load(open(os.path.join(dir_path, f'{algorithm}.pickle'), 'rb'))
    data = list(data.values())
    dataset = pd.DataFrame(dict(
        {f'var_{i}' : [data[j][0][i] for j in range(len(data))] for i in range(ALG_PARAMS[algorithm])},
        **{'quality' :  [0 if data[i][1] == 0 else -np.log(data[i][1]) for i in range(len(data))]}
        ))
    dataset['quality'] = dataset['quality'].replace(float('inf'), 100)

    dataset.to_csv(os.path.join(dir_path, f'{algorithm}_quality.csv'), index = False)

def train_quality_GBT(datasets_dir, estimators=1, max_depth=10, hyperparameter_search=False):
    
    for algorithm in ['saxpy', 'convolution', 'correlation', 'fwt']:
        dataset = pd.read_csv(os.path.join(datasets_dir, f'{algorithm}_quality.csv'))
        #calculate mean and standard deviation, add scaled 'x' and scaled 'y' to the dataframe
        mean_data = dataset.mean(axis=0)
        std_data = dataset.std(axis=0)
        dataset = (dataset - mean_data) / std_data

        X, y = dataset.loc[ :, [col for col in dataset.columns if col.startswith('var')]], dataset['quality']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        # dataset['set'] = ['train' if i in X_train.index else 'test' for i in range(len(dataset))]

        
        if not hyperparameter_search:

            model = GradientBoostingRegressor(random_state = 42, max_depth = max_depth, n_estimators=estimators, loss='huber')
            model = model.fit(X_train, y_train)
            pickle.dump(model, open(os.path.join("GBTs", f'{algorithm}_quality_GradientBoostingRegressor_{max_depth}_{estimators}'), 'wb'))
        
        else:
            model = GradientBoostingRegressor(random_state = 42, loss='huber')
            grid = { "max_depth":[5,10], "n_estimators":[1,10,15] }
            clf = GridSearchCV(model, grid)
            random_search = clf.fit(X_train, y_train)
            
            print(f"Best hyperparameters: {random_search.best_params_}")
            model = random_search.best_estimator_
            pickle.dump(model, open(os.path.join("GBTs", 
                f'{algorithm}_quality_GradientBoostingRegressor_{random_search.best_params_["max_depth"]}_{random_search.best_params_["n_estimators"]}'), 'wb'))

        print(f'{algorithm}_quality_GradientBoostingRegressor saved')

def train_targets_GBT(datasets_dir, estimators=1, max_depth=10, hyperparameter_search=False):    
    
    for algorithm in ['saxpy', 'convolution', 'correlation', 'fwt']:
        for hw in HW:
            for target in TARGET:
                dataset = pd.read_csv(os.path.join(datasets_dir, f'{algorithm}_{hw}.csv'))

                #calculate mean and standard deviation, add scaled 'x' and scaled 'y' to the dataframe
                mean_data = dataset.mean(axis=0)
                std_data = dataset.std(axis=0)
                dataset = (dataset - mean_data) / std_data

                y = dataset["memory"] if target == 'memory' else dataset["time"]
                X = dataset.drop(columns=[*TARGET])
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
                
                if not hyperparameter_search:

                    model = GradientBoostingRegressor(random_state = 42, max_depth = max_depth, n_estimators=estimators, loss='huber')
                    model = model.fit(X_train, y_train)
                    pickle.dump(model, open(os.path.join("GBTs", f'{algorithm}_{hw}_{target}_GradientBoostingRegressor_{max_depth}_{estimators}'), 'wb'))
                
                else:
                    model = GradientBoostingRegressor(random_state = 42, loss='huber')
                    grid = { "max_depth":[5,10], "n_estimators":[1,10,15] }
                    clf = GridSearchCV(model, grid)
                    random_search = clf.fit(X_train, y_train)
                    
                    print(f"Best hyperparameters: {random_search.best_params_}")
                    model = random_search.best_estimator_
                    pickle.dump(model, open(os.path.join("GBTs", 
                        f'{algorithm}_{hw}_{target}_GradientBoostingRegressor_{random_search.best_params_["max_depth"]}_{random_search.best_params_["n_estimators"]}'), 'wb'))
                
                print(f'{algorithm}_{hw}_{target}_GradientBoostingRegressor saved')
                

if __name__ == "__main__":
    ml_path = os.path.dirname(os.path.join(os.getcwd(), __file__, "ML"))
    
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "GBTs")):
        os.makedirs(os.path.join(os.path.dirname(__file__), "GBTs"))

    for algorithm in ['saxpy', 'convolution', 'correlation', 'fwt']:
        if not os.path.exists(os.path.join("datasets", f'{algorithm}_quality.csv')):
            create_dataset_quality(ml_path, algorithm)

    train_quality_GBT("datasets", hyperparameter_search=True)
    train_targets_GBT("datasets", hyperparameter_search=True)
    