import load
import process
import pybaseball
import sklearn

def test_LR(x, y, test_size):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=test_size, random_state=42)
    reg = sklearn.linear_model.LinearRegression().fit(X_train, y_train)
    score = reg.score(X_test, y_test)
    print("Sklearn score: ", score)
    y_pred = reg.predict(X_test)
    counter = 0.0
    for r in range(len(y_test)):
        max = np.argmax(y_pred[r])
        if y_test.iloc[r] != max:
            counter += 1
    reg_res = 1-counter/X_train.shape[0]
    print("Calculted Accuracy: ", reg_res)
    return reg

def test_KNN(x, y, test_size):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=test_size, random_state=42)
    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh.fit(X_train, y_train)
    # sklearn.model_selection.cross_val_score(neigh, X_test, y_test, cv=5)
    # score = neigh.score(X_test, y_test)
    # print("Sklearn score: ", score)
    y_pred = neigh.predict_proba(X_test)
    counter = 0.0
    for r in range(len(y_test)):
        max = np.argmax(y_pred[r])
        if y_test.iloc[r] != max:
            counter += 1
    neigh_res = 1-counter/X_train.shape[0]
    print("Calculted Accuracy: ", neigh_res)
    return neigh

def test_RF(x, y, test_size):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.3, random_state=42)
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    score = rf_classifier.score(X_test, y_test)
    print("Sklearn score: ", score)
    y_pred = rf_classifier.predict(X_test)
    counter = 0.0
    for r in range(len(y_test)):
        max = np.argmax(y_pred[r])
        if y_test.iloc[r] != max:
            counter += 1
    rf_res = 1-counter/X_train.shape[0]
    print("Calculted Accuracy: ", rf_res)
    return rf_classifier

def test_MLP(x, y, test_size):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=test_size, random_state=42)
    mlp_lbfgs_best = sklearn.neural_network.MLPClassifier(
            hidden_layer_sizes=[5],
            activation='relu',
            alpha=0.0001,
            max_iter=500, tol=1e-6,
            random_state=1,
            )
    # FOR GRID SEARCH:
    # activation : (‘identity’, ‘logistic’, ‘tanh’, ‘relu’)
    # parameters = {'alpha': np.logspace(0,1, num=10), 'solver': ('lbfgs', 'sgd', 'adam')}
    # clf = GridSearchCV(mlp_lbfgs_best, parameters, verbose=4)
    # clf.fit(X_train, y_train)
    # clf.cv_results_.keys()
    mlp_lbfgs_best.fit(X_train, y_train)
    score = mlp_lbfgs_best.score(X_test, y_test)
    print("Sklearn score: ", score)
    y_pred = mlp_lbfgs_best.predict_proba(X_test)
    counter = 0.0
    for r in range(len(y_test)):
        max = np.argmax(y_pred[r])
        if y_test.iloc[r] != max:
            counter += 1
    LBFSG_RELU_res = 1-counter/X_train.shape[0]
    print("Calculated Accuracy: ", LBFSG_RELU_res)
    return mlp_lbfgs