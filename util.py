# code=utf-8
from __future__ import print_function
from spectrum import *
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from hpelm import ELM
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def fit(X, Y, model="SVM"):
    # shuffle data
    tr_set = np.concatenate((Y, X), axis=1)
    np.random.shuffle(tr_set)
    x = tr_set[:, 1:]
    y = tr_set[:, 0].reshape(-1, 1)
    y = y.ravel()

    # split data into train and test part for evaluating model performance
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )

    if model == "SVM":
        tuned_parameters = [
            {
                "kernel": ["rbf"],
                "gamma": [10 ** x for x in range(-7, 8)],
                "C": [10 ** x for x in range(-5, 1)],
            },
            {"kernel": ["linear"], "C": [10 ** x for x in range(-5, 1)]},
        ]
        clf = GridSearchCV(svm.SVC(probability=True), tuned_parameters, cv=5)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)  # model performance

        means = clf.cv_results_["mean_test_score"]
        stds = clf.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print("Best parameters set found on training set:")
        print()
        print(clf.best_params_)

    if model == "ELM":
        label_encoder = LabelEncoder()
        y = y.reshape(-1, 1)
        y = label_encoder.fit_transform(y)

        onehot_encoder = OneHotEncoder(sparse=False)
        T_onehot_encoded = onehot_encoder.fit_transform(y.ravel().reshape(-1, 1))

        clf = ELM(x.shape[1], T_onehot_encoded.shape[1], classification="ml")
        clf.add_neurons(500, "rbf_l2")
        clf.train(x, T_onehot_encoded)
        y_pred = clf.predict(x).argmax(1)
        y_test = y

    return clf, accuracy_score(y_test, y_pred)


def predict(X, clf, model="SVM"):
    if model == "SVM":
        return (clf.predict(X), clf.predict_proba(X).max())
    if model == "ELM":
        tmp = clf.predict(X)
        return (tmp.argmax(1), tmp.max())


# def feat_gen_wrapper(data, **kwfeats):
#     windows = moving_window(data, WINDOW_LENGTH, WINDOW_INCREMENT)
#     feat = np.zeros(shape=(1, FEAT_NUM * CHANNELS))
#     for window in windows:
#         feat = np.concatenate((feat, feat_gen(window, **kwfeats)), axis=0)
#     return feat[1:, :]


def feat_gen(data, **kwfeats):
    """
    generate features of one observation
    :param data:        each column represents one channel of electrodes
    :param kwfeats:     key word for features
    :return:            ndarray with shape (1, X), X depends on features
    """
    if len(kwfeats.keys()) == 0:
        return cal_rms(data)

    feat = np.zeros(shape=(1, data.shape[1]))

    if "ar4c" in kwfeats.keys():
        feat = np.concatenate((feat, cal_ar4c(data)), axis=0)
    if "rms" in kwfeats.keys():
        feat = np.concatenate((feat, cal_rms(data)), axis=0)
    if "waveLength" in kwfeats.keys():
        feat = np.concatenate((feat, cal_waveLength(data)), axis=0)
    if "mav" in kwfeats.keys():
        feat = np.concatenate((feat, cal_mav(data)), axis=0)
    return feat[1:, :].reshape(1, -1)


def cal_rms(data):
    rms = np.zeros(shape=(1, data.shape[1]))
    for col in range(data.shape[1]):
        rms[:, col] = np.sqrt(np.mean(data[:, col] ** 2))
    return rms


def cal_waveLength(data):
    wl = np.zeros(shape=(1, data.shape[1]))
    for col in range(data.shape[1]):
        result = 0
        for i, j in zip(data[:-1, col], data[1:, col]):
            result += abs(j - i)
        wl[:, col] = result
    return wl


def cal_mav(data):
    mav = np.zeros(shape=(1, data.shape[1]))
    for col in range(data.shape[1]):
        mav[:, col] = np.mean(np.abs(data[:, col]))
    return mav


def cal_ar4c(data):
    # arma = ARMA(data.transpose(), order=(4, 0)).fit(disp=-1)
    arc = np.zeros(shape=(4, data.shape[1]))
    for col in range(data.shape[1]):
        arc_tmp, _, _ = aryule(data[:, col], 4)
        arc[:, col] = arc_tmp
    return arc


# def moving_window(x, window, step):
#     return [x[i : i + window] for i in range(0, (len(x) + 1) - window, step)]


# def generate(emg, reps, movements):
#     """
#     extract features of EMG based on certain reps of a given movements range
#     size of output: [len(windows), CHANNELS, FREQ_NUM] which is [Observations x CHANNELS x Feature Length]
#     """
#     # initiate data to concatenate
#     X = np.zeros((1, CHANNELS * FEAT_NUM))
#     Label = np.zeros((1, 1))
#
#     for mv in movements:
#         # training data
#         X_mv = np.zeros((1, CHANNELS * FEAT_NUM))
#         for rep in reps:
#             mask = np.where(np.logical_and(stimulus == mv, repetitions == rep))[0]
#             masked_emg = emg[mask, :]
#             X_rep = feat_gen_wrapper(masked_emg, waveLength=1, mav=1, ar4c=1)
#             X_mv = np.concatenate((X_mv, X_rep), axis=0)
#         observ_mv = X_mv.shape[0] - 1  # remove the first fake observation
#         X = np.concatenate((X, X_mv[1:, :]), axis=0)
#         label_mv = np.ones((observ_mv, 1)) * mv
#         Label = np.concatenate((Label, label_mv), axis=0)
#     return X[1:, :], Label[1:, :]


class Node:
    def __init__(self, data, node=None):
        self.data = data
        self.next = node


# if "__main__" == __name__:
# data = np.array([ 1, 7, 7, 8, 8, 7, 10, 3, 6, 7])
# print(feat_gen(data, mav=1))
# import h5py

# CHANNELS = 12
# FEAT_NUM = 6
# WINDOW_LENGTH = 400
# WINDOW_INCREMENT = 200
#
# # load dataset
# # data = elm.read("iris.data")
# raw = scipy.io.loadmat('D:/EMG/EMG_Signal/NinaPro_Database/DB2/DB2_s1/DB2_s1/S1_E1_A1.mat')
# emg = raw['emg']
# stimulus = raw['restimulus']
# repetitions = raw['rerepetition']
#
# movements = np.unique(stimulus)[1:]
# train_reps = np.array([1, 3, 4, 6])
# test_reps = np.array([2, 5])
#
# # These variables should be saved into disk
# X_train, Label_train = generate(emg, train_reps, movements)
# X_test, Label_test = generate(emg, test_reps, movements)
#
# f = h5py.File('test_data.h5', 'w')
# f.create_dataset('X_tr', data=X_train)
# f.create_dataset('Y_tr', data=Label_train)
# f.create_dataset('X_te', data=X_test)
# f.create_dataset('Y_te', data=Label_test)
# f.close()


# f = h5py.File('test_data.h5', 'r')
# X_train = f.get('X_tr').value
# Label_train = f.get('Y_tr').value
# X_test = f.get('X_te').value
# Label_test = f.get('Y_te').value

# tr_set = np.concatenate((Label_train,X_train), axis=1)
# te_set = np.concatenate((Label_test,X_test), axis=1)
#
# np.random.shuffle(tr_set)
# np.random.shuffle(te_set)
#
# X_train = tr_set[:,1:]
# Label_train = tr_set[:,0].reshape(-1,1)
#
# X_test = te_set[:,1:]
# Label_test = te_set[:,0].reshape(-1,1)

# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(Label_test.ravel())

# onehot_encoder = OneHotEncoder(sparse=False)
# T_onehot_encoded = onehot_encoder.fit_transform(Label_train)

# elm = ELM(X_train.shape[1], T_onehot_encoded.shape[1],classification='ml')
# elm.add_neurons(1000, 'rbf_l2')
# elm.train(X_train, T_onehot_encoded)
# y_pred = elm.predict(X_test).argmax(1)
# print(accuracy_score(y, y_pred))

# acc = []
# for n_neurons in range(100, 1501, 100):
#     elm.add_neurons(100, 'rbf_l2') # add 100 neurons every time
#     elm.train(X_train, T_onehot_encoded, "LOO")
#     y_pred = elm.predict(X_test).argmax(1)
#     acc_tmp = accuracy_score(y, y_pred)
#     acc.append(acc_tmp)
#     print(elm.nnet.get_neurons()[0][0], acc_tmp)
# max_acc = max(acc)
# max_idx = acc.index(max_acc)
# print(list(range(100, 1501, 100))[max_idx], max_acc)
