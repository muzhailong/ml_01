from scipy.stats import scoreatpercentile
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc


###result like result[['target','proba']]

def lift_curve_(result):
    result.columns = ['target', 'proba']

    result_ = result.copy()
    proba_copy = result.proba.copy()
    for i in range(10):
        point1 = scoreatpercentile(result_.proba, i * (100 / 10))
        point2 = scoreatpercentile(result_.proba, (i + 1) * (100 / 10))
        proba_copy[(result_.proba >= point1) & (result_.proba <= point2)] = ((i + 1))
    result_['grade'] = proba_copy
    df_gain = result_.groupby(by=['grade'], sort=True).sum() / (len(result) / 10) * 100
    plt.plot(df_gain['target'], color='red')
    for xy in zip(df_gain['target'].reset_index().values):
        plt.annotate("%s" % round(xy[0][1], 2), xy=xy[0], xytext=(-20, 10), textcoords='offset points')
        plt.plot(df_gain.index, [sum(result['target']) * 100.0 / len(result['target'])] * len(df_gain.index),
                 color='blue')
        plt.title('Lift Curve')
        plt.xlabel('Decile')
        plt.ylabel('Bad Rate (%)')
        plt.xticks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        plt.savefig("train.png")
        plt.show()


def roc_curve_(result):
    result.columns = ['target', 'proba']

    fpr, tpr, threshold = roc_curve(result['target'], result['proba'])  ###计算真正率和假正率  
    roc_auc = auc(fpr, tpr)  ###计算auc的值  
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc Curve')
    plt.legend(loc="lower right")
    plt.show()
