import pandas as pd
from tabulate import tabulate


def predict_mult_fun(self, *models, predict_data):
    """Multiple prediction function.

    Function takes user-defined models from models option list and
    multiple dimension array of defined values of parameters and
    calculate predicted values of piston cycle time for every
    specified surrogate model.

    Args:
        *models: Surrogate models argument list.
            Options:   "RFR" - Random Forrest Regression,
                        "LR" - Linear Regression,
                        "SVR" - Support Vector Regression,
                        "KNR" - K Nearest Neighbour Regression

        predict_data (array-like): multiple dimension array of parameter values
            to predict piston cycle time, by defined models in *models argument.
            Sub-array should have parameters organized as:
            ["M","S", "V_0", "k", "P_0", "T_a", "T_0"].

    Returns:
        Calculated predicted values of piston cycle time for every
        specified surrogate model.

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions
            that are relevant to the interface.
        ValueError: If `param2` is equal to `param1`.

    """
    prediction_df = pd.DataFrame()
    predict_data_df = pd.DataFrame(
        predict_data, columns=[
            "M", "S", "V_0", "k", "P_0", "T_a", "T_0"])

    for mm in models:
        mdl = self.models[mm]
        mdl.fit(self.train_X, self.train_y)
        pred = mdl.predict(predict_data)
        col_name = "Predicted val. - " + mm
        prediction_df[col_name] = pred
    dfs = [predict_data_df, prediction_df]
    pred_return = pd.concat(dfs, axis=1)
    print(tabulate(pred_return, headers="keys", tablefmt="psql"))
