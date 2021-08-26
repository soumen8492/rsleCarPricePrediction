# rsleCarPricePrediction
Prediction of Resale price of a car based on its old price, age, 1st hand or 2nd or n-hand,model name, Engine milege and other features.      
Applied Label Encoding and One hot Encoding techniques.     
Applied Supervised ML models to predict the result. We used SVM(Linear) to get maximum and more than 75% accuracy score.

"train-data-mdfdd.csv" file contains the data on which we applied our Model. It has the price,age, model name, Engine-fuel type and other necessary features.

In train.py file We used some encoding modules like LabelEncoder and OneHotEncoder for encoding the columns of the dataset to fit the model. We scaled the datasert using StandardScalar Module.

Then We used SVR(Support Vector Regressor) for the classification and prediction of the Resale Cars. After running the model many time it gave an average accuracy more than 75%.
