#!/usr/bin/env python3
import itertools
import numpy as np
import pandas as pd
from sklearn import tree, metrics
import pydotplus
import matplotlib.pyplot as plt


#FEATURES = ['genesis_basin', 'nature', 'track_type', 'lon', 'lat',
FEATURES = ['nature', 'track_type', 'month', 'lon', 'lat',
            'dist2land', 'msw', 'mcp']
LABEL = 'landfall'
#CATEGORY_FEATURES = ['genesis_basin', 'nature', 'track_type']
CATEGORY_FEATURES = ['nature', 'track_type', 'month']
CLASS_LABELS = ['nolandfall', 'landfall'] # [0, 1]


def extract_features(df):
    """Extract features."""
    return df[FEATURES]


def extract_label(df):
    """Extract class feature."""
    return df[LABEL].astype('int')


def get_dummies(df, categories):
    """Return data frame where categorical columns are replaced with
    dummy/indicator values."""
    return pd.get_dummies(df, prefix_sep='=', columns=categories)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


trainset_landfall_file = "tropicalstorms-trainingset-landfall.h5"
trainset_nolandfall_file = "tropicalstorms-trainingset-nolandfall.h5"
testset_file = "tropicalstorms-testset.h5"

# read in training set and test set
trainset_landfall = pd.read_hdf(trainset_landfall_file)
trainset_nolandfall = pd.read_hdf(trainset_nolandfall_file)
testset = pd.read_hdf(testset_file)

# join training sets
trainset = trainset_landfall.append(trainset_nolandfall)

#print(trainset_landfall.iloc[0])
#print(trainset_nolandfall.iloc[0])
#print(testset.iloc[0])
#print(trainset.iloc[0])

# train the classifier
X = get_dummies(extract_features(trainset), CATEGORY_FEATURES)
y = extract_label(trainset)
#print(X.info())
#print(X.values)
#print(y.values)
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=9)
clf = clf.fit(X, y)

# predict on test
X_test = get_dummies(extract_features(testset), CATEGORY_FEATURES)
y_test_truth = extract_label(testset).values
y_test_pred = clf.predict(X_test)
print(y_test_pred)
print(y_test_truth)

# print confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test_truth, y_test_pred)
print(cnf_matrix)
print(cnf_matrix.ravel())
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=CLASS_LABELS)
#plt.show()
plt.savefig('conf_matrix.png')

# print metrics
print("accuracy_score: %f" % metrics.accuracy_score(y_test_truth, y_test_pred))
print("average_precision_score: %f" % metrics.average_precision_score(y_test_truth, y_test_pred))
print("f1_score: %f" % metrics.f1_score(y_test_truth, y_test_pred))
print("recall_score: %f" % metrics.recall_score(y_test_truth, y_test_pred))
print("roc_auc_score: %f" % metrics.roc_auc_score(y_test_truth, y_test_pred))
print("classification_report:\n%s" % metrics.classification_report(y_test_truth, y_test_pred, target_names=CLASS_LABELS))

# print tree
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=X.columns.values,
                                class_names=CLASS_LABELS,
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('test.pdf')
