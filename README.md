
# Surrogate modeling of piston simulation function
[![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)  [![GitHub latest commit](https://badgen.net/github/last-commit/Naereen/Strapdown.js)](https://github.com/murko-a/Surrogate-model-piston-cycle/blob/master/src/piston_sim_fun_surrogate_murkoa/piston_fun_sm.py/commit/)  [![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)  [![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)


## Table of contents
- [Surrogate modeling of piston simulation function](#surrogate-modeling-of-piston-simulation-function)
  - [Table of contents](#table-of-contents)
  - [General info](#general-info)
  - [Requirements](#requirements)
  - [Setup](#setup)
  - [Code Examples](#code-examples)
  - [Support](#support)
  - [Roadmap](#roadmap)
  - [Authors and acknowledgment](#authors-and-acknowledgment)

## General info
<p align="justify"> This project is about surrogate modeling of piston simulation function that calculates cycle time of one piston rotation regarding multiple parameters. Project is based on a equations,</p>

$$ C(x) = 2\pi \sqrt{{M \over k+S^2 \frac{P_0 V_0}{T_0} \frac{T_a}{V^2} }} , \text{where} $$
$$ V = S \frac{S}{2k}\bigg(\sqrt{A^2 + 4k \frac{P_0 V_0}{T_0}T_a} -A \bigg)$$
$$A = P_0S + 19.62 M - \frac{kV_0}{S}$$	
Symbols:
* $M$ &nbsp; piston weight (kg)
* $S$ &nbsp; piston surface area (m$^2$)
* $V_0$ &nbsp; 	initial gas volume (m$^3$)
* $k$ &nbsp; spring coefficient (N/m)
* $P_0$ &nbsp; atmospheric pressure (N/m$^2$)
* $T_a$ &nbsp; ambient temperature (K)
* $T_0$ &nbsp; filling gas temperature (K)

<p align="justify"> Upper equation for calculating the cycle time <i>C(x)</i>, in which piston complete one cycle, is very complex and time expensive to calculate, therefore surrogate models for that equation were developed in this project. In project we have used following surrogate modeling methods: </p>

* Multiple Linear Regression,
* Random Forrest Regression,
* Support Vector Regression,
* K-Nearest Neighbors Regression.

Methods are in package functions used as:
<div align="center">

| Method                      | Package deffinition|
|:---------------------------:|:------------------:|
| Multiple Linear Regression           | "MLR"               |
| Random Forrest Regression   | "RFR"              |
| Support Vector Regression   | "SVR"              |
|K-Nearest Neighbors Regression   | "KNR"              |

</div >

<p align="justify"> To develop surrogate models of piston cycle function with specified methods, we have used <i>scikit-learn</i> package. Developed models are then used to calculate predict values. Performance of models is evaluated with <i>accuracy</i>, <i>mean absolute error (MAE)</i>, <i>mean square error (MSE)</i>, <i>root mean square error (RMSE)</i>, <i>R2</i> and <i>time</i>. Performance can be presented in values and also visually using <i>boxplot</i> or <i>lolipop</i> plot-type. True vs. predicted values can be obtained and also ploted with <i>line-scatter</i> plot. Also cycle time (true and predicted values) dependency of parameters can be observed visually with <i>line-scatter</i> plot. </p>

## Requirements 
This script requires the following modules:
 * [Numpy =  v1.22.4](https://numpy.org/)
 * [PyYAML = v6.0](https://pyyaml.org/)
 * [Pandas = v1.4.2](https://pandas.pydata.org/)
 * [Matplotlib = v3.5.2](https://matplotlib.org/)
 * [scikit-learn = v1.1.1](https://scikit-learn.org/stable/)
 * [SMT: Surrogate Modeling Toolbox = v1.2.0](https://smt.readthedocs.io/en/latest/)
	
## Setup
To run this project, install it locally using pip:

```
$ pip install piston_fun_sm
```

## Code Examples
Code to initially generate surrogate model:
 ```python
 from piston_fun_sm import SurrogateModel

 sm = SurrogateModel()

 # as argument is SruurogateModel class you can define your own parameter
 # space with just parameter limits (.yml file) or with all parameter values
 # (.csv file), as additional arguments you can use n_splt that defines
 # number of splits of data, shuff thats defines to shuffle data and
 # rand_state that defines random state --> for additional info check:
 # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

  
 # if you want to use your own parameter limit --> look .yml file in /data
 my_par_yml = "my_par.yml"
 sm_yml = SurrogateModel(my_par_yml, n_splt = 10, shuffle = True, rand_state = 42)

 # if you want to use your own parameter space --> look .csv file in /data
 my_par_csv = "my_par.csv"
 sm_csv = SurrogateModel(my_par_csv, n_splt = 10, shuffle = True, rand_state = 42)
 ```
 If you want to see folds that were created by KFold method and number of data values in these folds you can use:
 ```python
 sm.show_folds()
 ```
 Then with difined surrogate model we can use internal functions. For example quick prediction of your own data:
```python
 data = [30,0.005,0.002,1000,90000,292,345]
 sm.predict("RFR","SVR" ,"KNR", predict_data = data)
```
Than if you want to predict more values with defined parameter space, you can use:
```python
 mult_data = np.array([[30,0.005,0.002,1000,90000,292,345],
                    [30,0.015,0.008,2000,105000,293,355],
                    [30,0.009,0.004,4800,99000,296,348]])
 sm.predict_multiple("RFR","SVR" ,"KNR", predict_data = mult_data)
``` 
<p align="justify">If we want to get the performance evaluation of surrogate models and also plot it, we have two options of plot:</p>

* boxplot plot
* lolipop plot

```python
 #performance with boxplot plot
 sm.performance("RFR","MLR","KNR", "SVR", 
                perf_df=True, predict_data = None, plot_type="boxplot")

 #performance with lolipop plot
 sm.performance("RFR","MLR","KNR", "SVR", 
                perf_df=True, predict_data = None, plot_type="lolipop")

 #performance with lolipop plot
 sm.performance("RFR","MLR","KNR", "SVR", 
                perf_df=True, predict_data = None, plot_type="spyderweb")
```
If we want to compare results of true and predicted values and also plot it:
```python
# by enabling results argument function returns also dataframe of parameters,
# true and predicted values and by enabling plot argument, function returns
# plot of comparison
 sm.compare_true_pred("RFR","MLR","KNR", "SVR",
                     plot = True, results = True)
```
<p align="justify">If you want to compare true and predicted values dependent on specific parameter in defined parameter space you can use:</p>

```python
# by enabling results argument function returns also dataframe of parameters,
# true and predicted values
# to see panel plot of all parameter dependencies you can choose "all"
sm.param_true_pred("RFR","MLR","KNR", "SVR", 
                plot_type = "all", results = True)

# to see plot of specified parameter dependency you can choose between defined
# parameters ["M","S", "V_0", "k", "P_0", "T_a", "T_0"]
sm.param_true_pred("RFR","MLR","KNR", "SVR", 
                plot_type = "M", results = True)
```
## Support
<p align="justify">If you come to any of issues, have problems with the script or have some other questions about the script please send us e-mail on <a href = "mailto: anze.murko@rwth-aachen.com"> anze.murko@rwth-aachen.com </a>.</p>

## Roadmap
In the future we want to develop script that will use more different surrogate methods as:
* Artificial Neural Networks, 
* Gaussian Process,
* Bayesian Netorks, 
etc.

<p align="justify">to model defined piston cycle function. We would also like to develop used surrogate modeling methods to perform better and also like to enable more options in training, prediction, plotting functions, that will be user-defined.</p>


## Authors and acknowledgment
<p align="justify">We would like to express our appreciation to the professor Anil Yildiz and asistant Hu Zhao for supporting and helping us with this project. </p>

