#################################################################################################################
                      HADA: an Automated Tool for Hardware Dimensioning of AI Applications                     
#################################################################################################################
HADA is an optimization engine for hardware dimensioning and algorithm configuration, designed as the vertical 
matchmaking layer of the European AI on-demand platform.

In this version, HADA is designed for the Transprecision Computing use case: the engine receives a collection of
user-defined requirements on the execution of a given algorithm, and returns an optimal pair (hardware-platform, 
algorithm-configuration), to run such algorithm while respecting the user needs. 

    1. The supported algorithms are: saxpy, convolution, correlation, fwt
    2. The requirements (objective/constraints) can be defined in terms of four targets: 
        a. quality: quality of the solution produced by the algorithm, computed from the distance (error) 
           between such solution and the one obtained by using maximum precision;
        b. time: time of the algorithm execution;
        c. memory: peak memory usage of the algorithm execution;
        d. price: usage cost of the hardware 

Theoretically, HADA is a Combinatorial Optimization model made up of the following two components.
    1. An empirical, user-independent component: a dedicated predictive model for each pair (hw, target), used to
       estimate the performance of the algorithm, in terms of this target, while running on this hw platform. 
       Note: the target price does not need to be predicted: the value of this target is indicated directly by            the hardware provider.
    2. A declarative, user-dependent component: user-defined constraints and optimization criteria.

Practically, HADA can be used by invoking, from the command line, the script hada.py, which builds the 
optimization model according to the user input, hence it solves this model to deliver an optimal solution. 
Precisely, HADA can be executed as follows: 

    python3 hada_omlt.py \
            --help                <show help message>
            --data_folder         <folder with data for bounds/robustness coefficients (def: datasets)>\
            --models_folder       <folder with predictive models (def: DTs)> \
            --algorithm           <algorithm to execute> \
            --price_pc            <usage price of pc hardware (def: 7)> \
            --price_g100          <usage price of g100 hardware (def: 22)> \
            --price_vm            <usage price of vm hardware (def: 38)> \
            --objective           <objective to optimize> \
            --constraint_time     <user constraint for time> \
            --constraint_quality  <user constraint for quality> \
            --constraint_memory   <user constraint for memory> \
            --constraint_price    <user constraint for price> \
            --robust_factor       <confidence margin to introduce into the user constraints (def: None)> \
            --export_model        <when given, the final HADA model is exported to a .lp file> \
            --export_results      <when given, the obtained solution is exported to a .csv file> \
            --export_log          <when given, the log is exported to a .log file> 
            --train               <when given, the GBTs are retrained with the given max_depth, estimators>
            --max_depth           <max_depth hyperparameter for GBTs to use>
            --estimators           <number of estimators hyperparameter for GBTs to use>
