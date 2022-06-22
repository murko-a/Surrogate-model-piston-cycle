import numpy as np

def predict_fun(self, *mdls, predict_data):
    A = predict_data[4]*predict_data[1] + 19.62*predict_data[0] - (predict_data[3]*predict_data[2])/predict_data[1]
    V = predict_data[1]/(2*predict_data[3])*(np.sqrt(A**2 
                            + 4*predict_data[3]*(predict_data[4]*predict_data[2]*predict_data[5])/predict_data[6])
                                            - A)
    C = 2*np.pi * np.sqrt(predict_data[0]/(predict_data[3]
                                            + (predict_data[1]**2*predict_data[4]*predict_data[2]*predict_data[5])
                                            /(predict_data[6]*V**2)))
    print('True value: {t:8.2f}'.format(t = C), '\n--------------------------------------')
    predict_data = np.array([predict_data])
    predictions = list()
    for mm in mdls:
        mdl = self.models[mm]
        mdl.fit(self.train_X, self.train_y)
        pred = mdl.predict(predict_data)[0]
        predictions.append(pred)
        print("Model: {},  Predicted Value: {p:8.2f}".format(mm, p = pred))