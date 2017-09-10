
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import sklearn.metrics
from sklearn.model_selection import ShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import IPython
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO
from IPython.display import Image
import time
from sklearn.metrics import accuracy_score, f1_score
import pydotplus

import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
from sklearn.metrics import precision_recall_curve, auc

df = pd.read_csv("smote result.csv", encoding = "ISO-8859-1")
data_clean = df.dropna()

predictors = data_clean[[   'nWarnings', 'nAlarm',  'Friday',   'Monday',   'Saturday', 'Sunday',   'Thursday', 'Tuesday',  'Wednesday']]

targets = data_clean.departure

scaler = StandardScaler()


scaler.fit(df.drop('departure', axis = 1))

scaled_features = scaler.transform(df.drop('departure', axis = 1))

df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1])


# accuracy_list = []
# accuracy_list_decision = []
# fMeasure_list = []
# fMeasure_list_decision = []
recall_list = []
recall_list_desicion = []

presicion_list = []
presicion_list_desicion = []

y_real = []
y_proba = []
# f, axes = plt.subplots(1, 2, figsize=(10, 5))

kf = KFold(n_splits=5)
i = 1

for training, testing in kf.split(df['departure']):
    print ('xxxxxxxxxxxxxxxxxxxxxxxxxx Run {} xxxxxxxxxxxxxxxxxxxxxxxxxx'.format(i))

    pred_train = df_feat.ix[training]
    tar_train = df['departure'][training]
    pred_test = df_feat.ix[testing]
    tar_test = df['departure'][testing]
    knn = KNeighborsClassifier(n_neighbors = 4)
    knn.fit(pred_train, tar_train)
    pred = knn.predict(pred_test)
    # precision, recall, _ = precision_recall_curve(tar_test, pred[:,1])
    # lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
    # plt.step(recall, precision, label=lab)
    # y_real.append(tar_test)
    # y_proba.append(pred[:,1])
    
    print('----------------------------KNN Classification------------------------------------')
    # print("Accuracy is :")
    # print(sklearn.metrics.accuracy_score(tar_test,pred))
    # accuracy_list.append(sklearn.metrics.accuracy_score(tar_test,pred))
    print(confusion_matrix(tar_test,pred))
    print(classification_report(tar_test,pred))
    # fMeasure_list.append(f1_score(tar_test,pred, average="macro"))
    
    #----------------------------------------recall and precision---------------
    print("Recall is: ")
    print(sklearn.metrics.recall_score(tar_test,pred))
    recall_list.append(sklearn.metrics.recall_score(tar_test,pred))
    #precision_score
    print("Precision score is: ")
    print(sklearn.metrics.precision_score(tar_test,pred))
    presicion_list.append(sklearn.metrics.precision_score(tar_test,pred))

    #Decision tree code
    print('----------------------------Decision Tree------------------------------------')
    pred_train_decision = predictors.ix[training]
    # print (pred_train)
    tar_train_decision = targets[training]
    pred_test_decision = predictors.ix[testing]
    tar_test_decision = targets[testing]

    #Build model on training data
    classifier_decision=DecisionTreeClassifier()
    classifier_decision=classifier_decision.fit(pred_train_decision,tar_train_decision)

    predictions_decision=classifier_decision.predict(pred_test_decision)
    # precision, recall, _ = precision_recall_curve(tar_test, predictions_decision[:,1])
    # lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
    # plt.step(recall, precision, label=lab)
    # y_real.append(tar_test)
    # y_proba.append(predictions_decision[:,1])
    
    print(sklearn.metrics.confusion_matrix(tar_test_decision,predictions_decision))
    print(classification_report(tar_test_decision,predictions_decision))
    # accuracy_list_decision.append(sklearn.metrics.accuracy_score(tar_test_decision,predictions_decision))
    # fMeasure_list_decision.append(f1_score(tar_test_decision,predictions_decision, average="macro"))
    
    print("Recall is: ")
    print(sklearn.metrics.recall_score(tar_test_decision,predictions_decision))
    recall_list_desicion.append(sklearn.metrics.recall_score(tar_test_decision,predictions_decision))
    #precision_score
    print("Precision score is: ")
    print(sklearn.metrics.precision_score(tar_test_decision,predictions_decision))
    presicion_list_desicion.append(sklearn.metrics.precision_score(tar_test_decision,predictions_decision))    

    Displaying the decision tree

    out = StringIO()
    tree.export_graphviz(classifier_decision, out_file=out)
  
    graph=pydotplus.graph_from_dot_data(out.getvalue())
    #Create graph pdf 1 for each run
    millis = int(round(time.time() * 1000))  # Generate time system time in milliseconds
    #Image(graph.write_pdf("graph"+str(millis)+".pdf"))

    #Calculate accuracy

    print("Accuracy Score for graph"+str(millis)+".pdf is")
    print(sklearn.metrics.accuracy_score(tar_test_decision, predictions_decision))
    i+=1

    error_rate = []
    for i in range(1,20):
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(pred_train, tar_train)
        pred_i = knn.predict(pred_test)
        error_rate.append(np.mean(pred_i != tar_test))
    plt.figure(figsize=(10,6))
    plt.plot(range(1,20),error_rate, color = 'blue',linestyle = 'dashed', marker = 'o', markerfacecolor='red',markersize=10)
    plt.title('Error rate V/S K Value')
    plt.xlabel('k')
    plt.ylabel('error rate')
    plt.show()


print ("Accuracy for 5 folds KNN Classifier {}".format(sum(accuracy_list) / len(accuracy_list)))
print ("Accuracy for 5 folds decision tree {}".format(sum(accuracy_list_decision) / len(accuracy_list_decision)))
print ("F-Measure for 5 folds KNN Classifier {}".format(sum(fMeasure_list) / len(fMeasure_list)))
print ("F-Measure for 5 folds Decision tree {}".format(sum(fMeasure_list_decision) / len(fMeasure_list_decision)))
x = [1,2,3,4,5]


ax = plt.subplot(111)
plt.title('Accuracy Comparison')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')

bar1 = ax.bar([float(y)-.1 for y in x], accuracy_list,width=0.1,color='r',align='center')
bar2 = ax.bar(x, accuracy_list_decision,width=0.1,color='g',align='center')
ax.legend((bar1[0], bar2[0]), ('KNN Classifier', 'Decision tree'))
# ax.xaxis()
plt.show()

print('----------------------------recall comparison------------------------------------')
ax1 = plt.subplot(111)
plt.title('recall comparison')
plt.xlabel('Iteration')
plt.ylabel('recall')

bar3 = ax1.bar([float(y)-.1 for y in x], recall_list,width=0.1,color='r',align='center')
bar4 = ax1.bar(x, recall_list_desicion,width=0.1,color='g',align='center')
ax1.legend((bar3[0], bar4[0]), ('KNN Classifier', 'Decision tree'),loc=2)

print('----------------------------Precision comparison------------------------------------')
ax1 = plt.subplot(111)
plt.title('Precision comparison')
plt.xlabel('Iteration')
plt.ylabel('Precision')

bar3 = ax1.bar([float(y)-.1 for y in x], presicion_list,width=0.1,color='r',align='center')
bar4 = ax1.bar(x, presicion_list_desicion,width=0.1,color='g',align='center')
ax1.legend((bar3[0], bar4[0]), ('KNN Classifier', 'Decision tree'),loc=2)


plt.show()
combined_list = [accuracy_list]


# print('----------------------------precision_recall_curve------------------------------------')
y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
precision, recall, _ = precision_recall_curve(y_real, y_proba)
lab = 'Overall AUC=%.4f' % (auc(recall, precision))

print(precision)
plt.step(recall, precision, label=lab, lw=2, color='black')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left', fontsize='small')

#f.tight_layout()
plt.show()




