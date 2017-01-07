import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split

iris = pd.read_table('iris',sep=',')
iris.columns = ['a1','a2','b1','b2','label']
ss = ['a1','a2','b1','b2']
iris['label'][iris['label'] == 'Iris-setosa'] = 0
iris['label'][iris['label'] == 'Iris-virginica'] = 1
iris['label'][iris['label'] == 'Iris-versicolor'] = 2
print iris[0:10]
print iris['label'].drop_duplicates()
# split train and validation
train_x, val_x, train_y, val_y = train_test_split(iris[ss], iris.label,train_size=0.9,random_state=2016)
print len(train_x),len(val_x),len(train_y),len(val_y)
# xgbtrain = xgb.DMatrix(iris[ss],label=iris['label'])
# xgbtest = xgb.DMatrix(iris[ss],label=iris.label)
xgbtrain = xgb.DMatrix(train_x,train_y)
xgbtest = xgb.DMatrix(val_x,val_y)
watchlist = [(xgbtrain,'train'),(xgbtest,'val')]
# specify parameters via map
param = {
    'max_depth':1,
    'eta':0.01,
    'silent':1,
    'objective':'multi:softmax',
    'num_class':3}
num_round = 200
#model
bst = xgb.train(
    param,
    xgbtrain,
    num_round,
    watchlist,
    early_stopping_rounds=10)
# make prediction
preds = bst.predict(xgbtest)
print val_y.values
print preds
