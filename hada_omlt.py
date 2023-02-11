import argparse
import time
import pandas as pd
import pickle 
import warnings
import os
import csv
import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from omlt import OmltBlock
from omlt import scaling
from omlt.gbt.gbt_formulation import GBTBigMFormulation, add_formulation_to_block
from omlt.gbt.model import GradientBoostedTreeModel
from sklearn.ensemble import GradientBoostingRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from prettytable import PrettyTable
from ml import train_quality_GBT, train_targets_GBT
warnings.filterwarnings('ignore')


####### METADATA ########
HW = ["pc", "vm", "g100"]
TARGET = ["time", "quality", "memory", "price"]
ALG_PARAMS = {
        "convolution" : 4, 
        "saxpy" : 3,
        #"blackscholes" : 15, 
        "correlation" : 7, 
        "fwt" : 2 
        }

####### UTILITY FUNCTIONS #######
def logger(x, export_log):
    if export_log:
        with open("hada.log", "a") as file_log:
            file_log.write(f"{x}\n")
    print(x)


def extract_var_bounds(data_folder, algorithm, price):
    
    '''
    Compute upper and lower bounds of each variable 
    
    PARAMETERS
    ---------
    algorithm [string]: a transprecision computing algorithm {saxpy, convolution, correlation, fwt}
    data_folder [string]: path to the folder containing the necessary datasets 
    price [dict]: price of each hw platform  

    RETURN
    ------
    var_bounds [pd.DataFrame]: a frame with lower/upper bound for each variable
    '''

    bounds_min = {}
    bounds_max = {}
    for hw in HW:
        dataset = pd.read_csv(f"{data_folder}/{algorithm}_{hw}.csv")
        bounds_min[hw] = dataset.min()
        bounds_max[hw] = dataset.max()
    var_bounds = pd.DataFrame({
            "min" : pd.DataFrame(bounds_min).transpose().min(), 
            "max" : pd.DataFrame(bounds_max).transpose().max()
            })
    dataset = pd.read_csv(f"{data_folder}/{algorithm}_quality.csv")
    var_bounds.loc['quality', 'min'] = dataset['quality'].min()
    var_bounds.loc['quality', 'max'] = dataset['quality'].max()
    for i in range(ALG_PARAMS[algorithm]):
        var_bounds.loc["var_" + str(i), "max"] = 53
    var_bounds = var_bounds.append(pd.DataFrame(
        {"min": [min(price.values())], "max": [max(price.values())]}, 
        index = ["price"]))
    print(var_bounds)
    return var_bounds

def extract_robust_coeff(robust_factor, data_folder, algorithm, mlmodel_files, TARGET):
    
    '''
    Compute robustness coefficients for each predictive model, according to the specified robustness factor
    
    PARAMETERS
    ---------
    robust_factor [float]: confidence margin to introduce into the user constraints
    data_folder [string]: path to the folder containing the necessary datasets 
    algorithm [string]: a transprecision computing algorithm {saxpy, convolution, correlation, fwt}
    mlmodel_files [dict]: path to the predictive model corresponding to each pair (hw, target) 
    TARGET [list]: targets specified by the user

    RETURN
    ------
    robust_coeff [dict]: robustness coefficient for each predictive model
    '''

    if robust_factor or robust_factor == 0:
        robust_coeff = {}
        for target in TARGET: 
            # The target quality requires a single robustness coefficient
            if target == "quality":
                dataset = pd.read_csv(f"{data_folder}/{algorithm}_quality.csv")
                model = pickle.load(open(mlmodel_files[(f"{target}")], 'rb'))
                dataset[f'{target}_pred'] = model.predict(dataset[[col for col in dataset.columns if 'var' in col]])
                dataset[f'{target}_error'] = (dataset[f'{target}'] - dataset[f'{target}_pred']).abs()
                robust_coeff[(f"{target}")] = dataset[f'{target}_error'].std() * dataset[f'{target}_error'].quantile(robust_factor)
            else:
                for hw in HW: 
                    # The target price is not estimated: it does not require any robustness coefficient 
                    if target == 'price': 
                        robust_coeff[(f"{hw}", "price")] = 0
                    else: 
                        dataset = pd.read_csv(f"{data_folder}/{algorithm}_{hw}.csv")
                        model = pickle.load(open(mlmodel_files[(f"{hw}", f"{target}")], 'rb'))
                        dataset[f'{target}_pred'] = model.predict(dataset[[col for col in dataset.columns if 'var' in col]])
                        dataset[f'{target}_error'] = (dataset[f'{target}'] - dataset[f'{target}_pred']).abs()
                        robust_coeff[(f"{hw}", f"{target}")] = dataset[f'{target}_error'].std() * dataset[f'{target}_error'].quantile(robust_factor)
        return robust_coeff
    else:
        return None


def compute_mean_std(data_folder, algorithm, target, hw):
    dataset = pd.read_csv(f"{data_folder}/{algorithm}_{target if target == 'quality' else hw}.csv")
    mean_data = dataset.mean(axis=0)
    std_data = dataset.std(axis=0)
    
    y_mean = mean_data[target]
    y_std = std_data[target]
    X_mean = mean_data[[f"var_{i}" for i in range(ALG_PARAMS[algorithm])]]
    X_std = std_data[[f"var_{i}" for i in range(ALG_PARAMS[algorithm])]]
    
    return X_mean, X_std, y_mean, y_std

def get_scaling(algorithm, target, hw):
    '''Computes input bounds and the scaler object
    to feed the omlt model'''
    X_mean, X_std, y_mean, y_std = compute_mean_std("datasets", algorithm, target, hw)
    scaler = scaling.OffsetScaling(offset_inputs=X_mean.tolist(),
                    factor_inputs=X_std.tolist(),
                    offset_outputs=[y_mean],
                    factor_outputs=[y_std])
    input_bounds = {i:
                    ((var_bounds.loc[f"var_{i}", "min"] - X_mean.tolist()[i]) / X_std.tolist()[i],
                    (var_bounds.loc[f"var_{i}", "max"] - X_mean.tolist()[i]) / X_std.tolist()[i]) for i in range(ALG_PARAMS[algorithm])}
    return scaler, input_bounds


########### HADA #############
def HADA(algorithm, objective, user_constraint, price, var_bounds, mlmodel_files, export_log=False, robust_coeff=None):

    '''
    Implement HADA:
        1. Declare variables and basic constraints
        2. Embed predictive models 
        3. Declare user-defined constraints and objective 
        4. Solve the model and output an optimal matching (hw-platform, alg-configuration)

    PARAMETERS
    ---------
    algorithm [string] : a transprecision computing algorithm {saxpy, convolution, correlation, fwt}
    objective [dict]: type {min, max} and target
    user_constraint [dict]: type {'leq', 'geq', 'eq'} and right-hand side of each user-defined constraint  
    price [dict]: price of each hw platform  
    var_bounds [pd.DataFrame]: lower and upper bound for each target 
    mlmodel_files [dict]: path to the predictive model corresponding to each pair (hw, target) 
    export_log [bool]: when True, the log is written to hada.log
    robust_coeff [dict] : robustness coefficient for each pair (hw, target), for each performance target 
                          constrained by the user

    RETURN
    ------
    sol [dict]: optimal solution found
    model [pyomo.Model]: final optimization model
    '''

    logger("\n==============> Building", export_log)

    ####### MODEL #######
    start_time = time.time()

    # ALL_VARS = [f"b[{hw}]" for hw in HW] + \
    #            [f"y[{hw}_{target}]" for hw in HW for target in TARGET if target != "quality"] + [f"y[quality]"] + \
    #            [f"var[{i}]" for i in range(ALG_PARAMS[algorithm])]
    DECLARED_VARS = []

    model = pyo.ConcreteModel()

    ####### VARIABLES #######
    # Binary variable indexed by hw index set
    model.index_b = pyo.Set(initialize=HW)
    model.b = pyo.Var(model.index_b, domain=pyo.Binary)
    DECLARED_VARS += [f"b[{hw}]" for hw in HW]
    # Continuous variables representing the target values 
    target_index_set = []
    for target in set(list(user_constraint.keys()) + [objective['target']]):
        if target == "quality":
            target_index_set.append('quality')
            DECLARED_VARS.append("y[quality]")
        else:
            for hw in HW:
                target_index_set.append(f"{hw}_{target}")
                DECLARED_VARS.append(f"y[{hw}_{target}]")

    model.index_y = pyo.Set(initialize=target_index_set)
    # for indexed variables, bounds are defined through a function
    # that has access to the model and indexes
    def bound_target_var(model, i):
        if i != "quality":
            _, i = i.split("_")         
        return (var_bounds.loc[f"{i}", "min"], var_bounds.loc[f"{i}", "max"])
    
    model.y = pyo.Var(model.index_y, bounds=bound_target_var, domain=pyo.Reals)


    # An integer variable for each integer parameter of the algorithm, 
    # representing the value assigned to this parameter
    def bound_algorithm_var(model, i):
        return (var_bounds.loc[f"var_{i}", "min"], var_bounds.loc[f"var_{i}", "max"])


    model.index_var = pyo.Set(initialize=range(ALG_PARAMS[algorithm]))
    model.var = pyo.Var(model.index_var ,domain=pyo.Integers, bounds=bound_algorithm_var)
    model.aux_var = pyo.Var(model.index_var ,domain=pyo.Reals, bounds=bound_algorithm_var)
    DECLARED_VARS += [f"var[{i}]" for i in range(ALG_PARAMS[algorithm])]
    
    ####### CONSTRAINTS ######
    # HW Selection Constraint, enabling the selection of a single hw platform
    # model.index_hw_selection = pyo.Set(initialize=["hw_selection"])
    model.hw_selection = pyo.Constraint(expr=sum(model.b[hw] for hw in model.b) == 1)
    
    # Integrality Constraints, enabling the conversion of the auxiliary variables from continuous to integer
    model.eq_constraint = pyo.Constraint(model.index_var, rule=lambda model, i : model.var[i] == model.aux_var[i])
    
    logger("\nModel size before embedding model", export_log)
    logger(f"   #Variables: {model.nvariables()}", export_log) 
    logger(f"   #Constraints: {model.nconstraints()}", export_log)

    logger("\nStart embedding predictive models", export_log)
    
    ## EMPIRICAL CONSTRAINTS GRADIENT BOOSTED TREES ##

    for target in set(list(user_constraint.keys()) + [objective['target']]):
        # target price is not predicted, but indicated by the hw provider: it does not require any
        # dedicated predictive model
        if target == "price": 
            continue
       # target quality does not depend on the hw: it requires a unique predictive model
        elif target == "quality":
            setattr(model, f'block_{target}', OmltBlock())
            block = getattr(model, f'block_{target}') 


            gbt = pickle.load(open(mlmodel_files[f"{target}"], 'rb'))
            
            scaler, input_bounds = get_scaling(algorithm=algorithm, target=target, hw=None)
            # initial type defines type and length of model input variable 
            initial_type = [('input', FloatTensorType([None, ALG_PARAMS[algorithm]]))]
            onx = convert_sklearn(gbt, initial_types=initial_type)
            gbt_model = GradientBoostedTreeModel(onx, scaling_object=scaler, scaled_input_bounds=input_bounds)
            formulation = GBTBigMFormulation(gbt_model)
            
            block.build_formulation(formulation)
            block.connect_input = pyo.Constraint(model.index_var, rule=lambda block, i : model.aux_var[i] == block.inputs[i])
            block.connect_output = pyo.Constraint(expr=model.y[f'{target}'] == block.outputs[0])
            # add_formulation_to_block(block, gbt_model, block.aux_var, [model.y[target]])
            logger(f"   Predictive model corresponding to ({target}) embedded", export_log)
        # time and memory depend on both the hw and the algorithm configuration: each of them requires three 
        # dedicated predictive models
        else:
            for hw in HW:
                setattr(model, f'block_{hw}_{target}', OmltBlock())
                block = getattr(model, f'block_{hw}_{target}')

                gbt = pickle.load(open(mlmodel_files[(f"{hw}", f"{target}")], 'rb'))
                scaler, input_bounds = get_scaling(algorithm=algorithm, target=target, hw=hw)
                # initial type defines type and length of model input variable 
                initial_type = [('input', FloatTensorType([None, ALG_PARAMS[algorithm]]))]
                onx = convert_sklearn(gbt, initial_types=initial_type)
                gbt_model = GradientBoostedTreeModel(onx, scaling_object=scaler, scaled_input_bounds=input_bounds)
                formulation = GBTBigMFormulation(gbt_model)
                block.build_formulation(formulation)
                block.connect_input = pyo.Constraint(model.index_var, rule=lambda block, i : model.aux_var[i] == block.inputs[i])
                block.connect_output = pyo.Constraint(expr=model.y[f'{hw}_{target}'] == block.outputs[0])
                # add_formulation_to_block(block, gbt_model, block.aux_var, [model.y[f'{hw}_{target}']])
                logger(f"   Predictive model corresponding to ({hw}, {target}) embedded", export_log)

    #print(model.block_pc_price)
    logger("\nModel size after embedding empirical components", export_log)
    logger(f"   #Variables: {model.nvariables()}", export_log) 
    logger(f"   #Constraints: {model.nconstraints()}", export_log)
    
    
    logger("\nHardware prices:", export_log)
    for hw in HW: 
        logger(f"    {hw} = {price[hw]}", export_log)

    # Handling non-estimated target (price) and robustness coefficients: 
    # 1.Equality constraints, fixing each price variable y_hw_price to the usage price of the corresponding hw,
    # as required by the hw provider
    if 'price' in list(user_constraint.keys()) + [objective['target']]:
        model.index_constraint_price_hw = pyo.Set(initialize=[f"{hw}_price" for hw in HW])
        model.constraint_price_hw = pyo.Constraint(model.index_constraint_price_hw, rule=lambda model, i: model.y[i] == price[i.split("_")[0]])
    
    logger("\nUser requirements:", export_log)
    for target in user_constraint:
        logger(f"    {target} {user_constraint[target]['type']} {user_constraint[target]['bound']}", export_log)

    # 2. If no robustness is required, fix all coefficients to 0 
    if robust_coeff is None:
        robust_coeff = {(hw, target) : 0 for hw in HW for target in user_constraint.keys() if target != 'quality'}
        robust_coeff['quality'] = 0 

    logger("\nRoubstness Coefficients:", export_log)
    for key in robust_coeff.keys():
        logger(f"    {key} : {robust_coeff[key]}", export_log)

    # User-defined constraints, bounding the performance of the algorithm, as required by the user
    
    
    model.user_constraint = pyo.ConstraintList()
    for target in user_constraint.keys():
        if target == 'quality': 
            if user_constraint[target]["type"] == "leq":
                model.user_constraint.add(expr=model.y[target] <= user_constraint[f"{target}"]["bound"] - robust_coeff[(f'{target}')] )
            elif user_constraint[target]["type"] == "geq":
                model.user_constraint.add(expr=model.y[target] >= user_constraint[f"{target}"]["bound"] + robust_coeff[(f'{target}')])
            elif user_constraint[target]["type"] == "eq":
                model.user_constraint.add(expr=model.y[target] >= user_constraint[f"{target}"]["bound"] - robust_coeff[(f'{target}')])
                model.user_constraint.add(expr=model.y[target] <= user_constraint[f"{target}"]["bound"] + robust_coeff[(f'{target}')])
        else: 
            for hw in HW:
                # choose M as the smallest value that 
                # trivially satisfies the constraint 
                M = model.y[f"{hw}_{target}"].ub + user_constraint[f"{target}"]["bound"] + robust_coeff[(f'{hw}',f'{target}')]
                
                if user_constraint[target]["type"] == "leq":
                    model.user_constraint.add(
                            model.y[f"{hw}_{target}"] <= user_constraint[f"{target}"]["bound"] - robust_coeff[(f'{hw}',f'{target}')] + M*(1 - model.b[hw]))
                elif user_constraint[target]["type"] == "geq":
                    model.user_constraint.add(
                            model.y[f"{hw}_{target}"] >= user_constraint[f"{target}"]["bound"] + robust_coeff[(f'{hw}',f'{target}')] - M*(1 - model.b[hw]))
                elif user_constraint[target]["type"] == "eq":
                    model.user_constraint.add(
                            model.y[f"{hw}_{target}"] >= user_constraint[f"{target}"]["bound"] - robust_coeff[(f'{hw}',f'{target}')] - M*(1 - model.b[hw]))
                    model.user_constraint.add(
                            model.y[f"{hw}_{target}"] <= user_constraint[f"{target}"]["bound"] + robust_coeff[(f'{hw}',f'{target}')] + M*(1 - model.b[hw]))
    
    end_time = time.time()
    build_time = round(end_time - start_time, 2)
    logger(f"\n==> Building process terminated in {build_time}s", export_log)
    
    ##### OBJECTIVE #####
    logger("\nObjective", export_log)
    logger(f"    {objective['type']} {objective['target']}", export_log)

    if objective["target"] == "quality": 
        if objective["type"] == "min":
            model.objective = pyo.Objective(expr=model.y["quality"], sense=pyo.minimize)
        else: 
            model.objective = pyo.Objective(expr=model.y["quality"], sense=pyo.maximize)
    else: 
        if objective["type"] == "min":
            model.objective = pyo.Objective(expr=sum(model.y[f"{hw}_{objective['target']}"] * model.b[hw] for hw in HW), sense=pyo.minimize)
        else: 
            model.objective = pyo.Objective(expr=sum(model.y[f"{hw}_{objective['target']}"] * model.b[hw] for hw in HW), sense=pyo.maximize)
    

    solver = pyo.SolverFactory('cplex') # cplex needs compiled extension
    ##### SOLVE #####
    logger("\n==============> Solving", export_log)
    start_time = time.time()
    
    results = solver.solve(model, symbolic_solver_labels=True, tee=True)
    
    end_time = time.time()
    solve_time = round(end_time - start_time, 2)
    logger(f"\n==> Solving process terminated in {solve_time}s due to {results.solver.termination_condition}", export_log)


    if results.solver.termination_condition == TerminationCondition.optimal.name:

        
        solution = dict(
                {"objective" : (objective["type"], objective["target"])},
                **{target : (user_constraint[target]["type"], user_constraint[target]["bound"]) if target in user_constraint.keys() else "-" for target in TARGET}, 
                **{v.name : v.value for v in model.component_data_objects(pyo.Var) if v.name in DECLARED_VARS },
                **{"#variables" : model.nvariables()},
                **{"#constraints" : model.nconstraints()},
                **{"build_time" : build_time, "solve_time" : solve_time}
            )
    else:
        print('The following termination condition was encountered: ', results.solver.termination_condition)
        solution = dict(
               {"objective" : (objective["type"], objective["target"])},
               **{target : (user_constraint[target]["type"], user_constraint[target]["bound"]) if target in user_constraint.keys() else "-" for target in TARGET}, 
               **{v.name : None for v in model.component_data_objects(pyo.Var) if v.name in DECLARED_VARS},
               **{"#variables" : model.nvariables()},
               **{"#constraints" : model.nconstraints()},
               **{"build_time" : build_time, "solve_time" : solve_time}
               )
    model.write("sol.lp")
    return solution, model


if __name__ == "__main__":

    ####### INPUT #######
    parser = argparse.ArgumentParser(prog = 'HADA', description = 'An optimization engine for hardware dimensioning and algorithm configuration, designed as the vertical matchmaking\nlayer of the European AI on-demand platform.')
    parser.formatter_class = lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=55)
    parser.add_argument('--data_folder', type = str, default = "datasets", required = False,
            help = 'folder with data for bounds/robustness coefficients (def: datasets)')
    parser.add_argument('--models_folder', type = str, default = "GBTs", required = False,
            help = 'folder with predictive models (def: GBTs)')
    parser.add_argument('--algorithm', type = str, required = True,
            help = 'algorithm to execute', choices = ['saxpy', 'convolution', 'correlation', 'fwt'])
    # hw prices
    parser.add_argument('--price_pc', type = float, default = 7, required = False,
            help = 'usage price of pc hardware (def: 7)')
    parser.add_argument('--price_g100', type = float, default = 22, required = False,
            help = 'usage price of g100 hardware (def: 22)')
    parser.add_argument('--price_vm', type = float, default = 38, required = False,
            help =  'usage price of vm hardware (def: 38)')
    # objective
    parser.add_argument('--objective', type = str, required = True, nargs = 2,
            help =  'objective to optimize', metavar = ('{min,max}', 'TARGET'))
    # target bounds
    parser.add_argument('--constraint_time', type = str, required = False, nargs = 2,
            help =  'user constraint for time', metavar = ('{leq,eq,geq}', 'BOUND'))
    parser.add_argument('--constraint_quality', type = str, required = False, nargs = 2,
            help =  'user constraint for quality', metavar = ('{leq,eq,geq}', 'BOUND'))
    parser.add_argument('--constraint_memory', type = str, required = False, nargs = 2,
            help =  'user constraint for memory', metavar = ('{leq,eq,geq}', 'BOUND'))
    parser.add_argument('--constraint_price', type = str, required = False, nargs = 2,
            help =  'user constraint for price', metavar = ('{leq,eq,geq}', 'BOUND'))
    # robustness factor
    parser.add_argument('--robust_factor', type = float, default = None, required = False,
            help =  'confidence margin to introduce into the user constraints (def: None)')
    # model/results storing option
    parser.add_argument('--export_model', default=False, action='store_true',
            help =  'when given, the final HADA model is exported to a .lp file')
    parser.add_argument('--export_results', default=False, action='store_true',
            help =  'when given, the obtained solution is exported to a .csv file')
    parser.add_argument('--export_log', default=False, action='store_true',
            help =  'when given, the log is exported to a .log file')
    parser.add_argument('--train', default=False, action='store_true',
            help =  'when given, the GBTs are retrained with the given max_depth, estimators and loss hyperparameters')
    parser.add_argument('--max_depth', type=int, default=10, required=False,
            help =  'max_depth hyperparameter for GBTs')
    parser.add_argument('--estimators', type=int, default=1, required=False,
            help =  'number of estimators for GBTs')
    
    args = parser.parse_args()
    
    logger("=================================================================", args.export_log)
    logger(time.ctime(), args.export_log)
    logger(f"Running HADA for {args.algorithm} algorithm", args.export_log)

    if args.train:
        train_quality_GBT("datasets", estimators=args.estimators, max_depth=args.max_depth)
        train_targets_GBT("datasets", estimators=args.estimators, max_depth=args.max_depth)
    ##### TREES ######
    # load the trained empirical model to embed into HADA
    mlmodel_files = {(f"{hw}", f"{target}") : f"{args.models_folder}/{args.algorithm}_{hw}_{target}_GradientBoostingRegressor_{args.max_depth}_{args.estimators}" for hw in HW for target in TARGET if target != "quality"}
    mlmodel_files["quality"] = f"{args.models_folder}/{args.algorithm}_quality_GradientBoostingRegressor_{args.max_depth}_{args.estimators}"

    ###### USER DATA #####
    price = {"pc" : args.price_pc, "vm" : args.price_vm, "g100" : args.price_g100}
    user_constraint = {
            "price" : args.constraint_price,
            "time" : args.constraint_time, 
            "quality" : args.constraint_quality, 
            "memory" : args.constraint_memory 
            }
        
    user_constraint = {target : {"type" : constraint[0], "bound" : float(constraint[1])} 
            for target, constraint in user_constraint.items() if constraint is not None} 
    objective = {"type" : args.objective[0], "target" : args.objective[1]}
    
    ###### DATA #####
    # extract, from the training datasets of the empirical models, 
    # 1. upper/lower bound for each declarative variable 
    # 2. robustness coefficient for each predictive model
    var_bounds = extract_var_bounds(
            data_folder = args.data_folder, 
            algorithm = args.algorithm, 
            price = price)
    robust_coeff = extract_robust_coeff(
            robust_factor = args.robust_factor, 
            data_folder = args.data_folder, 
            algorithm = args.algorithm, 
            mlmodel_files = mlmodel_files,
            TARGET = list(user_constraint.keys()) + [objective['target']])

    # Build-and-Solve
    start_time = time.time()
    solution, model  = HADA(
            algorithm = args.algorithm, 
            objective = objective, 
            user_constraint = user_constraint,
            price = price,
            var_bounds = var_bounds, 
            mlmodel_files = mlmodel_files,
            robust_coeff = robust_coeff,
            export_log = args.export_log
            )
    end_time = time.time()
    solution["robust_factor"] = args.robust_factor
    
    # Print optimal solution    
    if solution["b[pc]"] is not None:
        logger("\n==============> Optimal solution found:", args.export_log)
        log_solution = pd.DataFrame({f"var[{i}]" : [None if solution[f"var[{i}]"] is None else round(solution[f"var[{i}]"])] for i in range(ALG_PARAMS[args.algorithm])})
        log_solution["hw"] = [round(solution["b[pc]"])*1 + round(solution["b[g100]"])*2 + round(solution["b[vm]"]*3)]
        log_solution["hw"] = log_solution["hw"].replace({1 : "pc", 2 : "g100", 3 : "vm"})
        for target in list(user_constraint.keys()) + [objective["target"]]:
                if target == "price":
                    log_solution[target] = [round(solution[f"y[{log_solution.loc[0, 'hw']}_{target}]"], 2)]
                elif target == "quality":
                    log_solution[f"pred {target}"] = [round(solution[f"y[{target}]"], 2)]
                else:
                    log_solution[f"pred {target}"] = [round(solution[f"y[{log_solution.loc[0, 'hw']}_{target}]"], 2)]

        log_table = PrettyTable()
        log_table.field_names = log_solution 
        log_table.add_row(log_solution.iloc[0, :])
        logger(f"\n{log_table}", args.export_log)
    
    else:
        logger("\n==============> No solution found", args.export_log)

    if args.export_results:
        # create fixed column names for csv
        # otherwise different executions will load wrong 
        # target names
        fieldnames =  ['objective']  
        fieldnames += [target for target in TARGET]
        fieldnames += [f"b[{hw}]" for hw in HW] 
        fieldnames += [f"y[{hw}_{target}]" for hw in HW for target in TARGET if target != "quality"] + ["y[quality]"] 
        fieldnames += [f"var[{i}]" for i in range(ALG_PARAMS[args.algorithm])]
        fieldnames += ["#variables", "#constraints", "build_time", "solve_time", "robust_factor"]
        
        filename = f"./{args.algorithm}_depth{args.max_depth}_est{args.estimators}_results.csv"
        if not os.path.exists(filename):
            
            # results = pd.DataFrame({k : [] for k in solution.keys()})
            with open(filename, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(solution)

            # results.to_csv(f"./{args.algorithm}_results.csv", index = False)

        else:
            with open(filename,'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                #writer=csv.writer(f)
                writer.writerow(solution)

        logger(f"\nResults exported into {filename}", args.export_log)

    if args.export_model:
        logger("\nExporting model", args.export_log)
        model.export_as_lp("hada.lp")
        logger("Model exported into hada.lp", args.export_log)
