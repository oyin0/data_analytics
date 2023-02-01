import pandas as pd
from scipy.stats import f_oneway
from sklearn.feature_selection import f_regression, SelectKBest
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse

car = pd.read_csv('CarSharing.csv', parse_dates=['timestamp'])

# dropping the duplicate rows and null values
car.drop_duplicates(inplace=True)
car.dropna(inplace=True)

# performing a one way ANOVA for the categorical variables
categories = ['season', 'holiday', 'workingday', 'weather']

for x in categories:
    # creating an array that groups into levels in each column when looping
    col = [car[car[x] == filtering]['demand'] for filtering in car[x].unique()]
    
    f_statistics, p_value = f_oneway(*col)
    print(f_statistics, p_value)
    
    if p_value < 0.05:
        print('There is a significant relationship between {} and demand.'.format(x))
    else:
        print('There is no significant relationship between {} and demand.'.format(x))


# performing a simple linear regression for the numerical variables
# defining the predictor and outcome variables
numeric = ['humidity', 'temp', 'temp_feel', 'windspeed']

y  = car['demand']
x = car[numeric]
test = SelectKBest(score_func = f_regression, k=4)

test.fit(x, y)

for i, (p_value) in enumerate(test.pvalues_):
    print('For {}, the p-value is {}'.format(numeric[i], p_value))
    
# changing timestamp column to datetime
car['timestamp'] = pd.to_datetime(car['timestamp'])

# set index for the time column
car.set_index('timestamp', inplace = True)

# subsetting the car dataframe to data only in 2017
car_2017 = car.loc['2017']

# subsetting temp data
temp_2017 = car_2017['temp']
# Multiplicative Decomposition 
result_mul = seasonal_decompose(temp_2017, model='multiplicative', extrapolate_trend = 0,
                                period = 1500)
# Additive Decomposition
result_add = seasonal_decompose(temp_2017, model='additive', extrapolate_trend=0, period = 1500)
# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
result_mul.plot().suptitle('Multiplicative Decompose')
result_add.plot().suptitle('Additive Decompose')
plt.show()

# subsetting humidity data
hum_2017 = car_2017['humidity']
# Additive Decomposition
result_add = seasonal_decompose(hum_2017, model='additive', extrapolate_trend=0, period = 1500)
# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
result_add.plot().suptitle('Additive Decompose')
plt.show()

# subsetting windspeed data
wind_2017 = car_2017['windspeed']
# Additive Decomposition
result_add = seasonal_decompose(wind_2017, model='additive', extrapolate_trend=0, period = 1500)
# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
result_add.plot().suptitle('Additive Decompose')
plt.show()

# subsetting demand data
demand_2017 = car_2017['demand']
# Additive Decomposition
result_add = seasonal_decompose(demand_2017, model='additive', extrapolate_trend=0, period = 1500)
# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
result_add.plot().suptitle('Additive Decompose')
plt.show()

# predicting weekly average demand rate using arima model
from statsmodels.tsa.arima_model import ARIMA
week = car['demand'].resample('W').mean()
split = int(len(week) * 0.3)
train_car, test_car = week[0:split], week[split:]

model = ARIMA(train_car, order=(2,1,2))
model_fit = model.fit()

pred = model_fit.forecast(len(test_car))[0]
print(pred)

# comparing between random forest with deep neural network
from sklearn.preprocessing import StandardScaler
num = car.select_dtypes(include=['float64', 'int64'])

scaler = StandardScaler()
X = num.drop(['demand'], axis=1)
y = num['demand']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# fit random forest
from sklearn.ensemble import RandomForestRegressor
randomf = RandomForestRegressor()
randomf.fit(X_train, y_train)
randomf_pred = randomf.predict(X_test)

# fit deep neural network
from sklearn.neural_network import MLPRegressor
deepn = MLPRegressor(hidden_layer_sizes=(50,50,50), max_iter=500, alpha=0.0001, solver='sgd', 
                     verbose=10, random_state=21, tol=0.000000001)
deepn.fit(X_train, y_train)
deepn_pred = deepn.predict(X_test)

# compare the performance 
from sklearn.metrics import mean_squared_error
mse_randomf = mean_squared_error(y_test, randomf_pred)
mse_deepn = mean_squared_error(y_test, deepn_pred)

print(f'MSE for Random Forest Regressor is {mse_randomf}')
print(f'MSE for Deep Neural Network is {mse_deepn}')


# categorizing the demand rate
import numpy as np
avg_demand = num['demand'].mean()

num['label'] = np.where(num['demand']>avg_demand, 1, 2)

X = num.drop(['demand', 'label'], axis=1)
y = num['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train, y_train)
log_pred = log.predict(X_test)

from sklearn.tree import DecisionTreeClassifier
dec = DecisionTreeClassifier()
dec.fit(X_train, y_train)
dec_pred = dec.predict(X_test)

from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# checking the accuracy of our models
from sklearn.metrics import accuracy_score
log_acc = accuracy_score(y_test, log_pred)
dec_acc = accuracy_score(y_test, dec_pred)
svm_acc = accuracy_score(y_test, svm_pred)

print(f'Logistic Regression accuracy score is {log_acc}')
print(f'Decision Tree Classifier accuracy score {dec_acc}')
print(f'SVC accuracy score is {svm_acc}')


# creating clusters using data in 2017 only 
from sklearn.cluster import KMeans

kmeans_2 = KMeans(n_clusters=2)
kmeans_3 = KMeans(n_clusters=3)
kmeans_4 = KMeans(n_clusters=4)
kmeans_12 = KMeans(n_clusters=12)

X = car_2017[['temp', 'temp_feel']]

kmeans_2.fit(X)
kmeans_3.fit(X)
kmeans_4.fit(X)
kmeans_12.fit(X)

clus_2 = kmeans_2.predict(X)
clus_3 = kmeans_3.predict(X)
clus_4 = kmeans_4.predict(X)
clus_12 = kmeans_12.predict(X)

from sklearn.metrics import silhouette_score
sil_2 = silhouette_score(X, clus_2)
sil_3 = silhouette_score(X, clus_3)
sil_4 = silhouette_score(X, clus_4)
sil_12 = silhouette_score(X, clus_12)

print(f'Silhouette score for k=2 is {sil_2}')
print(f'Silhouette score for k=3 is {sil_3}')
print(f'Silhouette score for k=4 is {sil_4}')
print(f'Silhouette score for k=12 is {sil_12}')




















    








