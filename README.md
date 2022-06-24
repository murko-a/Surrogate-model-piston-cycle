# Surrogate modeling of piston simulation function
[![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)  [![GitHub latest commit](https://badgen.net/github/last-commit/Naereen/Strapdown.js)](https://GitHub.com/Naereen/StrapDown.js/commit/)  [![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)  [![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)


## Table of contents
* [General info](#general-info)
* [Requirements ](#requirements)
* [Setup](#setup)
* [Code Examples](#code-examples)
* [Support](#support)
* [Roadmap](#roadmap)
* [Authors and acknowledgment](#authors-and-acknowledgment)

## General info
This project is about surrogate modeling of piston simulation function that calculates cycle of one piston rotation regarding multiple parameters.
	
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

 # or if you want to use your own parameter limit --> look .yml file in /data
 my_par_yml = 'my_par.yml'
 sm_yml = SurrogateModel(my_par_yml)

 # or if you want to use your own parameter space --> look .csv file in /data
 my_par_csv = 'my_par.csv'
 sm_csv = SurrogateModel(my_par_csv)
 ```
 Then with difined surrogate model we can use internal functions. For example predict our own data:
```python
 data = [30,0.005,0.002,1000,90000,292,345]
 sm.predict('RFR','SVR' ,'KNR', predict_data = data)
```
If we want to get the performance evaluation of surrogate models and also plot it, we have two options of plot:
* boxplot plot
* lolipop plot

```python
 #performance with boxplot plot
 sm.performance('RFR','SVR','KNR', 'SVR', perf_df=True, predict_data = None, plot_perf='boxplot')

 #performance with lolipop plot
 sm.performance('RFR','SVR','KNR', 'SVR', perf_df=True, predict_data = None, plot_perf='lolipop')
```
If we want to compare results of true and predicted values and also plot it:
```python
 sm.compare_true_pred('RFR','SVR', 'KNR', 'LR', plot = True, results = True)
```
## Support
If you come to any of issues, have problems with the script or have some other questions about the script please send us e-mail on anze.murko@rwth-aachen.com.

## Roadmap
In the future we want to develop script that will use more different surrogate methods as:
* Artificial Neural Networks, 
* Gaussian process,

to model defined piston cycle function. We would also like to develop used surrogate modeling methods to perform better. We would also like to enable more options in training, prediction, plotting functions, that will be user-defined.


## Authors and acknowledgment
We would like to express our appreciation to the professor Anil Yildiz and asistant Hu Zhao for supporting and helping us with this project. 