import csv
import os
import io
import time

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
import matplotlib.pyplot as plt
import pydotplus
import seaborn as sns

from IPython.core.pylabtools import figsize
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.tree import export_graphviz

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

"""THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

stores = os.path.join(THIS_FOLDER, 'static/stores.csv')
sales = os.path.join(THIS_FOLDER, 'static/sales_train.csv')
features = os.path.join(THIS_FOLDER, 'static/features.csv')"""

stores = pd.read_csv('static/stores.csv')  # (45,3)
#sales = pd.read_csv('static/sales.csv')
train_sales = pd.read_csv('static/sales_train.csv')  # (421 570, 5)
test_sales = pd.read_csv('static/sales_test.csv')  # (115 064, 4)
features = pd.read_csv('static/features.csv')  # (8 190, 12)

train_sales['Date'] = pd.to_datetime(train_sales['Date'])
test_sales['Date'] = pd.to_datetime(test_sales['Date'])
features['Date'] = pd.to_datetime(features['Date'])

# Объединяем в один датафрейм
# Попробовать через concat

# dataframe = pd.merge(stores, features, on=['Store'], how='left')
# dataframe = pd.merge(dataframe, sales, on=['Store', 'Date', 'IsHoliday'], how='left')

train_ = pd.merge(train_sales, stores)
train_dataframe = pd.merge(train_, features, on=['Store', 'Date', 'IsHoliday'], how='left')
test_ = pd.merge(test_sales, stores)
test_dataframe = pd.merge(test_, features, on=['Store', 'Date', 'IsHoliday'], how='left')

train_dataframe['Temperature'] = (train_dataframe['Temperature'] - 32) / 1.8000  # Фаренгейт в цельсий
test_dataframe['Temperature'] = (test_dataframe['Temperature'] - 32) / 1.8000  # Фаренгейт в цельсий
train_dataframe = train_dataframe.fillna(0)  # NaN на 0
test_dataframe = test_dataframe.fillna(0)  # NaN на 0

train_encode, train_type = train_dataframe['Type'].factorize()  # кодируем типы
train_dataframe['Type'] = train_encode
test_encode, test_type = test_dataframe['Type'].factorize()  # кодируем типы
test_dataframe['Type'] = test_encode

train_dataframe['IsHoliday'] = np.where((train_dataframe.IsHoliday == True), 1, 0)
test_dataframe['IsHoliday'] = np.where((test_dataframe.IsHoliday == True), 1, 0)

# корреляция
"""sns.set(rc={'figure.figsize':(20,10)}, font_scale=1.1)
sns.heatmap(train_dataframe.corr(), linewidths=0.5, annot=True, cmap='coolwarm')
plt.show()
sns.heatmap(test_dataframe.corr(), linewidths=0.5, annot=True, cmap='coolwarm')
plt.show()
"""

# графики

"""train_dataframe[['Date', 'Temperature']].plot(x='Date', figsize=(10, 10))
plt.show()
train_dataframe[['Date', 'Fuel_Price']].plot(x='Date', figsize=(10, 10))
plt.show()
train_dataframe[['Date', 'MarkDown1']].plot(x='Date', figsize=(10, 10))
plt.show()
train_dataframe[['Date', 'MarkDown2']].plot(x='Date', figsize=(10, 10))
plt.show()
train_dataframe[['Date', 'MarkDown3']].plot(x='Date', figsize=(10, 10))
plt.show()
train_dataframe[['Date', 'MarkDown4']].plot(x='Date', figsize=(10, 10))
plt.show()
train_dataframe[['Date', 'MarkDown5']].plot(x='Date', figsize=(10, 10))
plt.show()
train_dataframe[['Date', 'CPI']].plot(x='Date', figsize=(10, 10))
plt.show()
train_dataframe[['Date', 'Unemployment']].plot(x='Date', figsize=(10, 10))
plt.show()"""
"""
train_dataframe[
    ['Date', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI',
     'Unemployment']].plot(x='Date', figsize=(10, 10), subplots=True)
plt.show()

train_dataframe.groupby('Store').agg({'Weekly_Sales': "sum"}).reset_index().sort_values('Weekly_Sales',
                                                                                        ascending=False).plot(
    kind='bar')
plt.show()

plt.figure(figsize=(7, 5), dpi=65)
sns.set_style('ticks')
sns.barplot(y=train_dataframe["Weekly_Sales"], x=train_dataframe["Type"], palette='colorblind')
sns.despine()
plt.show()

sns.set_style('darkgrid')
sns.set(rc={'figure.figsize': (18, 7)})
sns.pointplot(x='Dept', y='Weekly_Sales', data=train_dataframe)
plt.grid()
plt.show()"""

# вытаскиваем из даты: день, неделю, месяц и год. Добавляем праздники
train_dataframe["Day"] = train_dataframe['Date'].dt.day
train_dataframe["Week"] = train_dataframe['Date'].dt.week
train_dataframe["Month"] = train_dataframe['Date'].dt.month
train_dataframe["Year"] = train_dataframe['Date'].dt.year

test_dataframe["Day"] = test_dataframe['Date'].dt.day
test_dataframe["Week"] = test_dataframe['Date'].dt.week
test_dataframe["Month"] = test_dataframe['Date'].dt.month
test_dataframe["Year"] = test_dataframe['Date'].dt.year

# супербоул
sb_1 = np.datetime64('2010-02-07')
sb_2 = np.datetime64('2011-02-06')
sb_3 = np.datetime64('2012-02-05')
sb_4 = np.datetime64('2013-02-03')

# день труда
labor_1 = np.datetime64('2010-09-06')
labor_2 = np.datetime64('2011-09-05')
labor_3 = np.datetime64('2012-09-03')
labor_4 = np.datetime64('2013-09-02')

# день благодарения
thx_1 = np.datetime64('2010-11-25')
thx_2 = np.datetime64('2011-11-25')
thx_3 = np.datetime64('2012-11-22')
thx_4 = np.datetime64('2013-11-28')

# рождество
xmas_1 = np.datetime64('2010-12-25')
xmas_2 = np.datetime64('2011-12-25')
xmas_3 = np.datetime64('2012-12-25')
xmas_4 = np.datetime64('2013-12-25')

train_dataframe['Superbowl'] = np.where(((train_dataframe.Date == sb_1) | (train_dataframe.Date == sb_2) |
                                         (train_dataframe.Date == sb_3) | (train_dataframe.Date == sb_4)), 1, 0)
test_dataframe['Superbowl'] = np.where(((test_dataframe.Date == sb_1) | (test_dataframe.Date == sb_2) |
                                        (test_dataframe.Date == sb_3) | (test_dataframe.Date == sb_4)), 1, 0)

train_dataframe['Labor'] = np.where(((train_dataframe.Date == labor_1) | (train_dataframe.Date == labor_2) |
                                     (train_dataframe.Date == labor_3) | (train_dataframe.Date == labor_4)), 1, 0)
test_dataframe['Labor'] = np.where(((test_dataframe.Date == labor_1) | (test_dataframe.Date == labor_2) |
                                    (test_dataframe.Date == labor_3) | (test_dataframe.Date == labor_4)), 1, 0)

train_dataframe['Thanksgiving'] = np.where(((train_dataframe.Date == thx_1) | (train_dataframe.Date == thx_2) |
                                            (train_dataframe.Date == thx_3) | (train_dataframe.Date == thx_4)), 1, 0)
test_dataframe['Thanksgiving'] = np.where(((test_dataframe.Date == thx_1) | (test_dataframe.Date == thx_2) |
                                           (test_dataframe.Date == thx_3) | (test_dataframe.Date == thx_4)), 1, 0)

train_dataframe['Xmas'] = np.where(((train_dataframe.Date == xmas_1) | (train_dataframe.Date == xmas_2) |
                                    (train_dataframe.Date == xmas_3) | (train_dataframe.Date == xmas_4)), 1, 0)
test_dataframe['Xmas'] = np.where(((test_dataframe.Date == xmas_1) | (test_dataframe.Date == xmas_2) |
                                   (test_dataframe.Date == xmas_3) | (test_dataframe.Date == xmas_4)), 1, 0)

train_dataframe['IsHoliday'] = train_dataframe['IsHoliday'] | train_dataframe['Superbowl'] | train_dataframe['Labor'] | \
                               train_dataframe['Thanksgiving'] | train_dataframe['Xmas']
test_dataframe['IsHoliday'] = test_dataframe['IsHoliday'] | test_dataframe['Superbowl'] | test_dataframe['Labor'] | \
                              test_dataframe['Thanksgiving'] | test_dataframe['Xmas']

drop_list = ['Superbowl', 'Labor', 'Thanksgiving', 'Xmas']
train_dataframe.drop(drop_list, inplace=True, axis=1)
test_dataframe.drop(drop_list, inplace=True, axis=1)

train_dataframe = pd.get_dummies(train_dataframe, drop_first=True)
test_dataframe = pd.get_dummies(test_dataframe, drop_first=True)

# корреляция

"""sns.set(rc={'figure.figsize': (20, 10)}, font_scale=1.1)
sns.heatmap(train_dataframe.corr(), linewidths=0.5, annot=True, cmap='coolwarm')
plt.show()

sns.set(rc={'figure.figsize': (20, 10)}, font_scale=1.1)
sns.heatmap(test_dataframe.corr(), linewidths=0.5, annot=True, cmap='coolwarm')
plt.show()"""

drop_list = ['MarkDown1', 'MarkDown4', 'Year', 'Month', 'Day', 'CPI', 'Unemployment']
train_dataframe.drop(drop_list, inplace=True, axis=1)
test_dataframe.drop(drop_list, inplace=True, axis=1)

for i in train_dataframe:
    if train_dataframe[i].dtypes == float:
        train_dataframe[i] = train_dataframe[i].astype(int)

for i in test_dataframe:
    if test_dataframe[i].dtypes == float:
        test_dataframe[i] = test_dataframe[i].astype(int)

train_X = train_dataframe.drop(['Date', 'Weekly_Sales'], axis=1)
train_y = train_dataframe['Weekly_Sales']

# print(train_dataframe.columns)

test_X = test_dataframe.drop('Date', axis=1).copy()

# print(train_X.shape, train_y.shape, test_X.shape)

"""
start_time = time.monotonic()

param_grid = {'n_estimators': np.arange(3, 101), 'max_depth': np.arange(2, 20)}
forest_reg = GridSearchCV(RandomForestRegressor(oob_score=False, warm_start=True, bootstrap=True, criterion='mse'),
                          param_grid, cv=5, n_jobs=-1, verbose=50)
forest_reg.fit(train_X, train_y)

end_time = time.monotonic()
print(datetime.timedelta(seconds=end_time - start_time))

print(forest_reg.best_params_)"""

"""
start_time = time.monotonic()
clf_rfr = RandomForestRegressor(n_estimators=23, n_jobs=-1, verbose=5, max_depth=13, warm_start=False)
clf_rfr.fit(train_X, train_y)

end_time = time.monotonic()
print('Время обучения: ')
print(timedelta(seconds=end_time - start_time))

kfold = KFold(n_splits=10, shuffle=True, random_state=7)
scores = cross_val_score(clf_rfr, train_X, train_y, cv=kfold)
print("Значения правильности перекрестной проверки: {}".format(scores))
print("Средняя правильность перекрестной проверки: {:.5f}".format(scores.mean()))

y_pred = clf_rfr.predict(test_X)

accuracy = round(clf_rfr.score(train_X, train_y) * 100, 2)
print('Accuracy of Random Forest model: ' + str(accuracy) + '%')
submission = pd.DataFrame({
    'Store': test_dataframe.Store.astype(str), 'Dept': test_dataframe.Dept.astype(str),
    'Date': test_dataframe.Date.astype(str),
    'Weekly_Sales_RFR': y_pred})
submission.to_csv('weekly_sales Predicted.csv', index=False)"""

"""dot_data = io.StringIO()
export_graphviz(clf_rfr.estimators_[13], out_file=dot_data, feature_names=list(train_X),
                filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('DecisionTree_13.pdf')
graph.write_png('DecisionTree_13.png')
"""

df = pd.read_csv('static/train_store_2.csv')
df_test = pd.read_csv('static/test_store_2.csv')

X = df.drop(['Weekly_Sales'], axis=1)
y_train = df['Weekly_Sales']

X_train = X.drop(['Date'], axis=1)

X_test = df_test.drop('Date', axis=1).copy()

# X_t, X_te, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# X_train = X_t.drop(['Date'], axis=1)
# X_test = X_te.drop(['Date'], axis=1)

"""start_time = time.monotonic()

param_grid = {'n_estimators': np.arange(3, 40), 'max_depth': np.arange(2, 13)}
forest_reg = GridSearchCV(RandomForestRegressor(oob_score=False, warm_start=True, bootstrap=True),
                          param_grid, cv=2, n_jobs=-1, verbose=50)
forest_reg.fit(X, y)

end_time = time.monotonic()
print('Время обучения: ')
print(timedelta(seconds=end_time - start_time))

print(forest_reg.best_params_)"""

start_time = time.monotonic()
clf_rfr = RandomForestRegressor(n_estimators=30, n_jobs=-1, verbose=5, max_depth=13, warm_start=False, random_state=6)
clf_rfr.fit(X_train, y_train)

end_time = time.monotonic()
print('Время обучения: ')
print(timedelta(seconds=end_time - start_time))

kfold = KFold(n_splits=10, shuffle=True, random_state=7)
scores = cross_val_score(clf_rfr, X_train, y_train, cv=kfold)
print("Значения правильности перекрестной проверки: {}".format(scores))
print("Средняя правильность перекрестной проверки: {:.5f}".format(scores.mean()))

y_pred = clf_rfr.predict(X_test)

"""print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

accuracy = round(clf_rfr.score(X_test, y_test) * 100, 2)
print('Accuracy of Random Forest model: ' + str(accuracy) + '%')
"""

submission_store1 = pd.DataFrame({
    'Dept': X_test.Dept.astype(str), 'Date': df_test.Date.astype(str), 'Weekly_Sales_RFR': y_pred})
submission_store1.to_csv('weekly_sales Predicted store1.csv', index=False)

"""dot_data = io.StringIO()
export_graphviz(clf_rfr.estimators_[13], out_file=dot_data, feature_names=list(X_train),
                filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('1.pdf')
graph.write_png('1.png')"""

df.set_index('Date', inplace=True)
submission_store1.set_index('Date', inplace=True)

sales = df.groupby('Date')['Weekly_Sales'].sum()
print(sales)
sales_pred = submission_store1.groupby('Date')['Weekly_Sales_RFR'].sum()
print(sales_pred)

plt.figure(figsize=(12, 7))
plt.plot(sales)
plt.xlabel('Years')
plt.ylabel('Weekly_Sales')
plt.show()

plt.figure(figsize=(12, 7))
plt.plot(sales_pred)
plt.xlabel('Years')
plt.ylabel('Weekly_Sales')
plt.show()

"""for i in range(1, 13):
    if [train_dataframe['Store'] == i]:
        store = train_dataframe.loc[train_dataframe['Store'] == i]
        del store['Store']
        number = str(i)
        store.to_csv('train_store' + '_' + number + '.csv', index=False, header=True)

for i in range(1, 13):
    if [test_dataframe['Store'] == i]:
        store = test_dataframe.loc[test_dataframe['Store'] == i]
        del store['Store']
        number = str(i)
        store.to_csv('test_store' + '_' + number + '.csv', index=False, header=True)
"""