import numpy as np


def predict_fun(self, *models, predict_data):
    """Quick prediction function.

    Function takes user-defined models from models option list and list
    of defined values of parameters and calculate true and predicted values of
    piston cycle time for every specified surrogate model.

    Args:
        *models: Surrogate models argument list.
            Options:   "RFR" - Random Forrest Regression,
                        "MLR" - Multiple Linear Regression,
                        "SVR" - Support Vector Regression,
                        "KNR" - K Nearest Neighbour Regression

        predict_data (list): List of parameter values to predict piston
            cycle time, by defined models in *models argument. List of
            parameters should be organized as:
            ["M","S", "V_0", "k", "P_0", "T_a", "T_0"].

    Returns:
        Calculated true and predicted values of piston cycle time.

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions
            that are relevant to the interface.
        ValueError: If `param2` is equal to `param1`.

    """
    A = predict_data[4] * predict_data[1] + 19.62 * predict_data[0] - \
        (predict_data[3] * predict_data[2]) / predict_data[1]
    V = predict_data[1] / (2 * predict_data[3]) * (np.sqrt(A**2 + 4 * predict_data[3] * (
        predict_data[4] * predict_data[2] * predict_data[5]) / predict_data[6]) - A)
    C = 2 * np.pi * np.sqrt(predict_data[0] / (predict_data[3] + (predict_data[1]**2 *
                            predict_data[4] * predict_data[2] * predict_data[5]) / (predict_data[6] * V**2)))
    print("True value: {t:8.2f}".format(t=C),
          "\n--------------------------------------")
    predict_data = np.array([predict_data])
    predictions = list()
    for mm in models:
        mdl = self.models[mm]
        mdl.fit(self.train_X, self.train_y)
        pred = mdl.predict(predict_data)[0]
        predictions.append(pred)
        print("Model: {},  Predicted Value: {p:8.2f}".format(mm, p=pred))
