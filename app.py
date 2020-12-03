import logging

from flask import Flask, render_template
from flask_cors import CORS

from datetime import datetime
import pandas as pd
import numpy as np
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

stores_dictionary = {
    'store1': 'static/train_store_1.csv',
    'store2': 'static/train_store_2.csv',
    'store3': 'static/train_store_3.csv',
    'store4': 'static/train_store_4.csv',
    'store1_test': 'static/test_store_1.csv',
    'store2_test': 'static/test_store_2.csv',
    'store3_test': 'static/test_store_3.csv',
    'store4_test': 'static/test_store_4.csv',

    'store5': 'static/train_store_5.csv',
    'store6': 'static/train_store_6.csv',
    'store7': 'static/train_store_7.csv',
    'store8': 'static/train_store_8.csv',
    'store5_test': 'static/test_store_5.csv',
    'store6_test': 'static/test_store_6.csv',
    'store7_test': 'static/test_store_7.csv',
    'store8_test': 'static/test_store_8.csv',

    'store9': 'static/train_store_9.csv',
    'store10': 'static/train_store_10.csv',
    'store11': 'static/train_store_11.csv',
    'store12': 'static/train_store_12.csv',
    'store9_test': 'static/test_store_9.csv',
    'store10_test': 'static/test_store_10.csv',
    'store11_test': 'static/test_store_11.csv',
    'store12_test': 'static/test_store_12.csv',
}

stores_list = []

class Stores:
    def __init__(self, store, csv_file_train, csv_file_test):
        self.store = store
        self.file_train = csv_file_train
        self.file_test = csv_file_test
        self.df_train = pd.read_csv(self.file_train)
        self.df_test = pd.read_csv(self.file_test)
        self.X_train = self.df_train.drop(['Weekly_Sales', 'Date'], axis=1)
        self.y_train = self.df_train['Weekly_Sales']

        self.X_test = self.df_test.drop(['Date'], axis=1).copy()
        self.regressor = RandomForestRegressor(n_estimators=30, n_jobs=-1, verbose=5, max_depth=13, warm_start=False,
                                               random_state=6)
        self.regressor.fit(self.X_train, self.y_train)

    def getPredictedValue(self):
        y_pred = self.regressor.predict(self.X_test)
        submission_store = pd.DataFrame({
            'Dept': self.X_test.Dept.astype(str), 'Date': self.df_test.Date.astype(str), 'Weekly_Sales_RFR': y_pred})

        # sales = self.df_train.groupby('Date')['Weekly_Sales'].sum()
        sales_pred = submission_store.groupby('Date')['Weekly_Sales_RFR'].sum().reset_index()
        # sales_pred = submission_store.set_index('Date').groupby['Weekly_Sales_RFR'].sum()

        # submission_store.to_csv('weekly_sales Predicted store1.csv', index=False)
        # sales_pred.to_frame(name='Sales').reset_index()
        return sales_pred


def getValues(name):
    # predict = pd.DataFrame()
    store = stores_list[0]
    for i in stores_list:
        if name == i.store:
            store = i
            break
    current_predict = store.getPredictedValue()
    # predict.append(current_predict)
    return current_predict


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/store/<name>')
def store_profile(name):
    forecast_values = getValues(name)

    forecast_x = forecast_values['Date']
    forecast_x = forecast_x.values.tolist()

    forecast_y = forecast_values['Weekly_Sales_RFR']
    forecast_y = forecast_y.values.tolist()

    context = {
        "name": name,
        "forecast_x": forecast_x,
        "forecast_y": forecast_y
    }
    return render_template('store.html', context=context)


if __name__ == '__main__':
    store1 = Stores('store1', stores_dictionary['store1'], stores_dictionary['store1_test'])
    stores_list.append(store1)
    store2 = Stores('store2', stores_dictionary['store2'], stores_dictionary['store2_test'])
    stores_list.append(store2)
    store3 = Stores('store3', stores_dictionary['store3'], stores_dictionary['store3_test'])
    stores_list.append(store3)
    store4 = Stores('store4', stores_dictionary['store4'], stores_dictionary['store4_test'])
    stores_list.append(store4)
    store5 = Stores('store5', stores_dictionary['store5'], stores_dictionary['store5_test'])
    stores_list.append(store5)
    store6 = Stores('store6', stores_dictionary['store6'], stores_dictionary['store6_test'])
    stores_list.append(store6)
    store7 = Stores('store7', stores_dictionary['store7'], stores_dictionary['store7_test'])
    stores_list.append(store7)
    store8 = Stores('store8', stores_dictionary['store8'], stores_dictionary['store8_test'])
    stores_list.append(store8)
    store9 = Stores('store9', stores_dictionary['store9'], stores_dictionary['store9_test'])
    stores_list.append(store9)
    store10 = Stores('store10', stores_dictionary['store10'], stores_dictionary['store10_test'])
    stores_list.append(store10)
    store11 = Stores('store11', stores_dictionary['store11'], stores_dictionary['store11_test'])
    stores_list.append(store11)
    store12 = Stores('store12', stores_dictionary['store12'], stores_dictionary['store12_test'])
    stores_list.append(store12)

    app.run()
