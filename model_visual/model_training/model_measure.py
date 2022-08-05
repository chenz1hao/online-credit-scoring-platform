import os

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# AUC图像绘制
from model_visual.settings import BASE_DIR


def AUC_plot(task_id, algorithmName, test_y, pred_y_prob):
    # print(algorithmName, "AUC图像绘制：")
    fpr, tpr, thresholds = roc_curve(test_y, pred_y_prob)
    auc = roc_auc_score(test_y, pred_y_prob)
    plt.plot(fpr, tpr)
    plt.title(algorithmName+" auc_plot=%.4f" % (auc))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.fill_between(fpr, tpr, where=(tpr > 0), color='green', alpha=0.5)

    os.makedirs(os.path.join(str(BASE_DIR) + '\\static\\auc_plot\\task_id_' + str(task_id)))
    plt.savefig(os.path.join(str(BASE_DIR) + '\\static\\auc_plot\\task_id_' + str(task_id)) + "\\auc.png")
    plt.clf() # 清除画板，防止重复绘制

# 输出打印算法性能
def getPerformance(task_id, algorithm_name, test_y, pred_y, pred_y_prob):
    # TP(True Positive) 预测正确的1
    # FN(False Negative) 预测为-1，真实为1
    # FP(False Positive) 预测为1，真实为-1
    # TN（True Negative) 预测为-1，真实为-1

    TP = []
    FN = []
    FP = []
    TN = []
    # 如果label是-1和1 / 0和1 不一样的是

    for i in range(len(pred_y)):
        if pred_y[i] == 1 and test_y[i] == 1:
            TP.append(i)
        elif pred_y[i] == 0 and test_y[i] == 1:
            FN.append(i)
        elif pred_y[i] == 1 and test_y[i] == 0:
            FP.append(i)
        elif pred_y[i] == 0 and test_y[i] == 0:
            TN.append(i)

    accuracy = (len(TP)+len(TN))/(len(TP)+len(FP)+len(TN)+len(FN))
    precision = len(TP) / (len(TP) + len(FP))
    recall = len(TP) / (len(TP) + len(FN))
    F1_score = 2 * ((precision*recall)/(precision+recall))
    AUC_plot(task_id, algorithm_name, test_y, pred_y_prob)
    return accuracy, precision, recall, F1_score