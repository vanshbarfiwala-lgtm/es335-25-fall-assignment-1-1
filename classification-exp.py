import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

X_df = pd.DataFrame(X, columns=['X1', 'X2'])
y_s = pd.Series(y, dtype='category')

#Q2)A
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_df, y_s, test_size=0.3, random_state=42)
tree = DecisionTree(criterion ='information_gain', max_depth=5)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

print("Accuracy:", accuracy(y_pred, y_test))

for cls in y_test.unique():
    print("Class",cls," Precision:", precision(y_pred, y_test, cls))
    print("Class",cls," Recall:", recall(y_pred, y_test, cls))

#Q2)B

k = 5
n = len(X_df)
dum = n // k
acc_list = []
depth_list = []
for i in range(k):
    start = i * dum
    end = (i+1) * dum if i < k-1 else n
    x_test = X_df.iloc[start:end]
    y_test = y_s.iloc[start:end]
    x_train = pd.concat([X_df.iloc[:start], X_df.iloc[end:]])
    y_train = pd.concat([y_s.iloc[:start], y_s.iloc[end:]])
    n2 = len(x_train)
    dum2 = n2 // k
    acc_per_depth = {1: [], 2: [], 3:[],4:[],5:[],6:[],7:[],8:[],9:[], 10: []}
    for j in range(k):
        start2 = j * dum2
        end2 = (j+1) * dum2 if j < k-1 else n2
        x_val = x_train.iloc[start2:end2]
        y_val = y_train.iloc[start2:end2]
        x_train2 = pd.concat([x_train.iloc[:start2], x_train.iloc[end2:]])
        y_train2 = pd.concat([y_train.iloc[:start2], y_train.iloc[end2:]])
        for x in range(10):
            tree = DecisionTree(criterion='information_gain', max_depth=x+1)
            tree.fit(x_train2, y_train2)
            y_pred = tree.predict(x_val)
            acc = accuracy(y_pred, y_val)
            acc_per_depth[x+1].append(acc)
    mean_acc_per_depth = {d: np.mean(accs) for d, accs in acc_per_depth.items()}
    best_depth = max(mean_acc_per_depth, key=mean_acc_per_depth.get)
    tree = DecisionTree(criterion='information_gain', max_depth=best_depth)
    tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test)
    acc = accuracy(y_pred, y_test)
    depth_list.append(best_depth)
    acc_list.append(acc)
    print(f"Best depth for fold {i+1}: {best_depth}")
    print(f"Accuracy using depth {best_depth} for fold {i+1}: {acc}")
print("Overall best depth of model:",max(depth_list,key = depth_list.count))
print("Mean accuray:",sum(acc_list)/k)