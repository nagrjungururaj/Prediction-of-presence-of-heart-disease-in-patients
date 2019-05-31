import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

classes = 2
classifier = ['forest','bayes','logit']

#read the data using pandas
dataframe = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath('heart.csv')),'heart.csv'))

#check for any missing/Nan/empty values
if dataframe.isnull().values.any() == True:
    print('Missing/Nan/empty values present')
else:
    print('No missing values found')

#convert the values to numpy array
data = dataframe.values

#Data and labels
X = data[:,0:data.shape[1]-1]
labels = data[:,-1]

#Split data with 85% train and 15% test
x_train, x_test, y_train, y_test = train_test_split(X,labels,test_size=0.15)

# SVM model

def get_classifier(classifier):

    if classifier == 'forest':
        model = RandomForestClassifier(n_estimators=100)
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        return y_pred
    else:
        if classifier == 'bayes':
            model = GaussianNB()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            return y_pred
        else:
            model = LogisticRegression()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            return y_pred
acc = []
f1 = []

for i in range(len(classifier)):
    y_pred = get_classifier(classifier[i])
    print('Results for classifier:',classifier[i])
    print('Accuracy =',accuracy_score(y_test,y_pred))
    print('F1-score =',f1_score(y_test,y_pred))

    acc.append(accuracy_score(y_test,y_pred))
    f1.append(f1_score(y_test,y_pred))

    c = confusion_matrix(y_test.astype(float), y_pred.round())
    sns.heatmap(c, annot=True, xticklabels=['Healthy','Heart-Disease'], yticklabels=['Healthy','Heart-Disease'])
    plt.title('Confusion matrix of samples for presence of heart disease')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

#bar plot to compare classifier perfromance
N = 3
ind = np.arange(N)
width = 0.5
p1 = plt.bar(ind, acc, width)
p1[0].set_color('r')
p1[1].set_color('g')
p1[2].set_color('b')
plt.title('Accuracy comparison for the classifiers used')
plt.ylabel('Accuracy in %')
plt.xticks(ind,('Random-forest', 'Naive-Bayes', 'Logistic-Regression'))
plt.show()

p2 = plt.bar(ind, f1, width)
p2[0].set_color('r')
p2[1].set_color('g')
p2[2].set_color('b')
plt.ylabel('F1-score in %')
plt.title('F1-score comparison for the classifiers used')
plt.xticks(ind,('Random-forest', 'Naive-Bayes', 'Logistic-Regression'))
plt.show()