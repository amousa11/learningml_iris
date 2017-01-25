from sklearn import svm
import csv

x = []
y = []

# Extract data from CSV
with open("Iris.csv") as cc:
    data = csv.DictReader(cc)
    for row in data:
        x.append([row["SepalLengthCm"],row["SepalWidthCm"],row["PetalLengthCm"],row["PetalWidthCm"]])
        y.append(row["Species"])

# Partition the data for training and testing

output_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

x = [[float(j) for j in feature] for feature in x]
# Convert to String->float
y = [output_map[output] for output in y]
# Convert to String->Int (0, 1, 2)

zeros, ones, twos = [(j, k) for j, k in zip(x, y) if k == 0], [(j, k) for j, k in zip(x, y) if k == 1], [(j, k) for j, k in zip(x, y) if k == 2]

x_train = [k[0] for k in zeros[:-10]] + [k[0] for k in ones[:-10]] + [k[0] for k in twos[:-10]]
y_train = [k[1] for k in zeros[:-10]] + [k[1] for k in ones[:-10]] + [k[1] for k in twos[:-10]]

x_test = [k[0] for k in zeros[-10:]] + [k[0] for k in ones[-10:]] + [k[0] for k in twos[-10:]]
y_test = [k[1] for k in zeros[-10:]] + [k[1] for k in ones[-10:]] +[k[1] for k in twos[-10:]]

# SVM

clf = svm.SVC(decision_function_shape='ovo', kernel='rbf')
clf.fit(x_train, y_train)

print("SVM Training Data score out of 1: " + str(clf.score(x_train, y_train)))
print("SVM Testing Data score out of 1: " + str(clf.score(x_test, y_test)))

# KNN

from sklearn import neighbors

knn = neighbors.KNeighborsClassifier(n_neighbors=5, p=2)
knn.fit(x_train, y_train)
print("KNN Training Data score out of 1: " + str(knn.score(x_train, y_train)))

print("KNN Testing data score out of 1: " + str(knn.score(x_test, y_test)))

# NN

from sklearn import neural_network

nn = neural_network.MLPClassifier()

nn.fit(x_train, y_train)
print("Neural network score out of 1: " + str(nn.score(x_train, y_train)))
print("Neural network score out of 1: " + str(nn.score(x_test, y_test)))
