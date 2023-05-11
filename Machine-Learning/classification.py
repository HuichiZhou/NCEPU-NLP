from data4ml import read_dataset, MyDataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

def nbayes():
    inputs, y_train = read_dataset('data/train.txt')
    ds = MyDataset()
    MAX_VOCAB_SIZE = 1000
    ds.set_stopword('data/scu_stopwords.txt')
    ds.build_vocab(inputs, MAX_VOCAB_SIZE)
    
    # 训练
    X_train = ds.transform(inputs)
    clf_nb = MultinomialNB()
    clf_nb.fit(X_train, y_train)

    # 测试  
    test, y_test = read_dataset('data/test.txt')
    X_test = ds.transform(test)
    y_pred = clf_nb.predict(X_test)
    with open('data/train_data.pkl', 'wb') as f:
        data = (X_train, y_train)
        pickle.dump(data, f)

    with open('data/test_data.pkl', 'wb') as f:
        data = (X_test, y_test)
        pickle.dump(data, f)
    print(accuracy_score(y_test, y_pred))

def logistic():
    # 训练
    with open('data/train_data.pkl', 'rb') as pkf:   # 导入已保存的训练数据和测试数据
        X_train, y_train = pickle.load(pkf)
    with open('data/test_data.pkl', 'rb') as pkf:
        X_test, y_test = pickle.load(pkf)

    clf_lr = LogisticRegression(penalty='l2',C=1,solver='saga')  # 模型参数可以根据分类结果进行调优
    clf_lr.fit(X_train, y_train)  # 模型训练

    y_pred = clf_lr.predict(X_test)  # 模型预测
    # 查看各类指标
    print(accuracy_score(y_test, y_pred))

if __name__ == '__main__':
    print("贝叶斯结果如下")
    nbayes()
    print("逻辑回归结果如下")
    logistic()