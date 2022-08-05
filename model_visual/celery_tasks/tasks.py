from django.utils import timezone
import os

from sklearn.model_selection import train_test_split
import joblib
import data_tool.generateOnehot
import opt_binning.get_binning
from data_tool import generate_data_set_tool
from model_visual import celery_app
from model_visual.settings import BASE_DIR
from mysite import models
from celery import platforms
from model_training import model_algorithm, model_measure
import pandas as pd
import numpy as np
platforms.C_FORCE_ROOT = True
import scorecardpy as sc

from nimbusml.linear_model import LogisticRegressionBinaryClassifier
from nimbusml import Pipeline


# 训练模型的异步任务
@celery_app.task
def train_process(task_id, model_id, data_set_id, train_test_split, random_state, label_name):
    model_name = models.model_class.objects.get(id=model_id).model_name
    data = models.data_set.objects.get(id=data_set_id)
    data_url = os.path.join(str(BASE_DIR) + '\\data\\' + data.data_file_name)

    if model_name == "logistic_regression":
        try:
            print('LogisticRegression异步训练开始')


            # 保存训练结果为pkl文件
            feature_name_list = list(pd.read_csv(data_url).columns)
            feature_name_list.remove(label_name)

            lr, test_X, test_y = model_algorithm.logistic_regression(data_url, train_test_split, random_state, label_name, feature_name_list)
            pred_y = lr.predict(test_X)
            pred_y_prob = lr.predict_proba(test_X)[:, 1]
            if not os.path.exists(os.path.join(str(BASE_DIR) + '\\task_results\\task_id_' + str(task_id))):
                os.makedirs(os.path.join(str(BASE_DIR) + '\\task_results\\task_id_' + str(task_id)))
            joblib.dump(lr, os.path.join(str(BASE_DIR) + '\\task_results\\task_id_' + str(task_id)) + '\\task_id_' + str(task_id) + ".pkl")


            # 保存模型性能评估信息
            df = pd.read_csv(data_url)
            features = df.columns.values[1:]
            train_result = {}
            count = 0
            for f in features:
                train_result[f] = lr.coef_[0][count]
                count = count + 1
            train_result['intercept'] = lr.intercept_[0]
            accuracy, precision, recall, f1_score = model_measure.getPerformance(task_id, model_name+"_TaskID_"+str(task_id), test_y, pred_y, pred_y_prob)

            # 将任务的模型类型设置为线性
            cur_task = models.task.objects.get(id=task_id)
            cur_task.model_if_linear = True
            cur_task.save()

        except Exception as e:
            cur_task = models.task.objects.get(id=task_id)
            cur_task.task_status = 2  # 2:训练失败
            cur_task.task_error_log = e
            cur_task.save()
            return "Logistic Regression Failed"

    elif model_name == "risk_slim":
        print('risk_slim异步训练开始')
        pred_y, pred_y_prob, test_y, solutions = model_algorithm.risk_slim(data_url, train_test_split, random_state)
        accuracy, precision, recall, f1_score = model_measure.getPerformance(task_id,
                                                                             model_name + "_TaskID_" + str(task_id),
                                                                             test_y, pred_y, pred_y_prob)
        train_result = "待完善"

    models.linear_task_result.objects.create(
        task_id=models.task.objects.get(id=task_id),
        task_result_str=str(train_result),
        task_result_url='/task_results/task_id_' + str(task_id) + "/task_id_" + str(task_id) + ".pkl",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        auc_plot_url='/static/auc_plot/task_id_' + str(task_id) + "/auc.png"
    )

    # 更新当前训练任务的训练状态与结束时间
    cur_task = models.task.objects.get(id=task_id)
    cur_task.task_end = timezone.now()
    cur_task.task_status = 1  # 1:训练成功
    cur_task.save()

    return "Train Success"


# 分箱与生成ONT_HOT编码的异步任务
@celery_app.task
def data_set_bin_and_one_hot(data_id, data_url, data_file_name):
    df = pd.read_csv(data_url)
    cols = [column for column in df]
    target = cols[0]
    features = cols[1:]
    var_split_list = {}
    for f in features:
        split = opt_binning.get_binning.binning(f, df[f], df[target])
        count = 1
        split_format = []
        for s in split:
            if count == 1:
                split_format.append("(-INF," + str(s) + "]")
            else:
                split_format.append("(" + str(split[count - 2]) + "," + str(split[count - 1]) + "]")
            count = count + 1

        split_format.append("(" + str(split[-1]) + ",+INF)")

        var_split_list[f] = split_format
        models.data_set_bin.objects.create(
            data_set_id=models.data_set.objects.get(data_url=data_url),
            feature_name=f,
            bin_result=split_format
        )


    data_tool.generateOnehot.generateOneHotByList(data_url, features, var_split_list,
                                                  "./data/one_hot/" + data_file_name)
    models.data_set.objects.filter(id=data_id).update(if_one_hot_finish = True)


@celery_app.task
def data_set_bin(binning_task_id, data_id, label_select, train_ratio, random_state, binning_feature_list, special_code):
    binning_task = models.binning_task.objects.get(id=binning_task_id)
    data = models.data_set.objects.get(id=data_id)
    data_url = data.data_url
    df = pd.read_csv(data_url)
    X = df.drop([label_select], axis=1)
    y = df[label_select]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-int(train_ratio))/100, random_state=int(random_state) )

    X_y_train = pd.concat([X_train, y_train], axis=1, ignore_index=False)
    bins = None
    if special_code != '':
        special_code = special_code.split(',')
        print(special_code)
        bins = sc.woebin(X_y_train, y=label_select, x=binning_feature_list, special_values=special_code)
    else:
        bins = sc.woebin(X_y_train, y=label_select, x=binning_feature_list)

    for f in binning_feature_list:
        this_feature_bins = bins[f]
        for i in range(len(this_feature_bins)):
            models.binning_feature.objects.create(
                binning_task_id = binning_task,
                data_set_id = data,
                feature_name = f,
                bin_num = len(this_feature_bins),
                which_bin = i+1,
                this_bin_str = this_feature_bins.iloc[i]['bin'],
                count = this_feature_bins.iloc[i]['count'],
                count_percent = this_feature_bins.iloc[i]['count_distr'],
                non_event_num = this_feature_bins.iloc[i]['good'],
                event_num = this_feature_bins.iloc[i]['bad'],
                event_rate = this_feature_bins.iloc[i]['badprob'],
                woe = this_feature_bins.iloc[i]['woe'],
                this_iv = this_feature_bins.iloc[i]['bin_iv'],
                total_iv = this_feature_bins.iloc[i]['total_iv'],
                is_special_values = this_feature_bins.iloc[i]['is_special_values'],
                breaks = this_feature_bins.iloc[i]['breaks'],
            )

    # 将bins pkl文件保存到本地
    if not os.path.exists(os.path.join(str(BASE_DIR) + '\\binning_result\\binning_task_' + str(binning_task_id))):
        os.makedirs(os.path.join(str(BASE_DIR) + '\\binning_result\\binning_task_' + str(binning_task_id)))
    joblib.dump(bins, os.path.join(str(BASE_DIR) + '\\binning_result\\binning_task_' + str(binning_task_id)) + '\\binning_task_' + str(binning_task_id) + ".pkl")

    # 更新当前分箱任务的状态与结束时间 以及 pkl文件
    binning_task.pkl_file_url = './binning_result/binning_task_' + str(binning_task_id) + '/'
    binning_task.task_end = timezone.now()
    binning_task.task_status = 1  # 1:训练成功
    binning_task.save()

    return "Binning Finished"

@celery_app.task
def update_dataset_bin(binning_task_id, binning_dict, special_code, binning_feature_list):
    binning_task = models.binning_task.objects.get(id = binning_task_id)
    data = binning_task.data_set_id
    data_url = data.data_url
    random_state = binning_task.random_state
    train_ratio = binning_task.train_ratio
    label_select = binning_task.training_label

    # 先删除掉这个binning_task_id的binning_feature
    models.binning_feature.objects.filter(binning_task_id=binning_task_id).delete()

    df = pd.read_csv(data_url)
    X = df.drop([label_select], axis=1)
    y = df[label_select]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - int(train_ratio)) / 100,
                                                        random_state=int(random_state))
    count = 0
    for i in special_code:
        special_code[count] = int(i)
        count += 1


    X_y_train = pd.concat([X_train, y_train], axis=1, ignore_index=False)

    bins = sc.woebin(X_y_train, y=label_select, x=binning_feature_list, breaks_list=binning_dict, special_values=special_code)
    for f in binning_feature_list:
        this_feature_bins = bins[f]
        for i in range(len(this_feature_bins)):
            models.binning_feature.objects.create(
                binning_task_id=binning_task,
                data_set_id=data,
                feature_name=f,
                bin_num=len(this_feature_bins),
                which_bin=i + 1,
                this_bin_str=this_feature_bins.iloc[i]['bin'],
                count=this_feature_bins.iloc[i]['count'],
                count_percent=this_feature_bins.iloc[i]['count_distr'],
                non_event_num=this_feature_bins.iloc[i]['good'],
                event_num=this_feature_bins.iloc[i]['bad'],
                event_rate=this_feature_bins.iloc[i]['badprob'],
                woe=this_feature_bins.iloc[i]['woe'],
                this_iv=this_feature_bins.iloc[i]['bin_iv'],
                total_iv=this_feature_bins.iloc[i]['total_iv'],
                is_special_values=this_feature_bins.iloc[i]['is_special_values'],
                breaks=this_feature_bins.iloc[i]['breaks'],
            )
    # 将bins pkl文件保存到本地
    if not os.path.exists(os.path.join(str(BASE_DIR) + '\\binning_result\\binning_task_' + str(binning_task_id))):
        os.makedirs(os.path.join(str(BASE_DIR) + '\\binning_result\\binning_task_' + str(binning_task_id)))
    joblib.dump(bins, os.path.join(str(BASE_DIR) + '\\binning_result\\binning_task_' + str(binning_task_id)) + '\\binning_task_' + str(binning_task_id) + ".pkl")

    binning_task.pkl_file_url = './binning_result/binning_task_' + str(binning_task_id) + '/'
    binning_task.last_modified = timezone.now()
    binning_task.task_status = 1
    binning_task.save()

    return "Binning Updated"

@celery_app.task
def create_score_card(train_task_id, binning_task_id, scorecard_feature_list):
    binning_task = models.binning_task.objects.get(id = binning_task_id)
    train_task = models.task.objects.get(id = train_task_id)
    pkl_file_url = binning_task.pkl_file_url
    bins = joblib.load(pkl_file_url + 'binning_task_' + str(binning_task_id) + ".pkl")
    random_state = binning_task.random_state
    train_ratio = binning_task.train_ratio
    label_name = binning_task.training_label
    data_file_name = binning_task.data_set_id.data_file_name

    # 创建评分卡
    ## 将原数据集用WoE值替换
    woe_data_url = "./data/woe/" + data_file_name
    dat = pd.read_csv(binning_task.data_set_id.data_url)

    woe = sc.woebin_ply(dat, bins)
    woe.to_csv(woe_data_url, index=False)
    ## 修改scorecard_feature_list （加"_woe")
    scorecard_feature_list = [feature + "_woe" for feature in scorecard_feature_list]
    ## 训练逻辑回归
    lr, test_X, test_y = model_algorithm.logistic_regression(woe_data_url, train_ratio, random_state, label_name, scorecard_feature_list)
    ## 评分卡 输出评分卡pkl
    card = sc.scorecard(bins, lr, scorecard_feature_list)
    if not os.path.exists(os.path.join(str(BASE_DIR) + '\\score_card\\score_card_id_' + str(train_task_id))):
        os.makedirs(os.path.join(str(BASE_DIR) + '\\score_card\\score_card_id' + str(train_task_id)))
    joblib.dump(card, './score_card/score_card_id_' + str(train_task_id) + ".pkl")

    ## 保存lr pkl
    if not os.path.exists(os.path.join(str(BASE_DIR) + '\\task_results\\task_id_' + str(train_task_id))):
        os.makedirs(os.path.join(str(BASE_DIR) + '\\task_results\\task_id_' + str(train_task_id)))
    joblib.dump(lr, os.path.join(str(BASE_DIR) + '\\task_results\\task_id_' + str(train_task_id)) + '\\task_id_' + str(train_task_id) + ".pkl")

    # 保存模型性能评估信息
    pred_y = lr.predict(test_X)
    pred_y_prob = lr.predict_proba(test_X)[:, 1]
    train_result = {}
    count = 0
    for f in scorecard_feature_list:
        train_result[f] = lr.coef_[0][count]
        count += 1
    train_result['intercept'] = lr.intercept_[0]
    accuracy, precision, recall, f1_score = model_measure.getPerformance(train_task_id,
                                                                         "score_card(logistic reression on WoE)" + "_TaskID_" + str(train_task_id),
                                                                         test_y,
                                                                         pred_y,
                                                                         pred_y_prob)
    models.linear_task_result.objects.create(
        task_id=models.task.objects.get(id=train_task_id),
        task_result_str=str(train_result),
        task_result_url='/task_results/task_id_' + str(train_task_id) + "/task_id_" + str(train_task_id) + ".pkl",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        auc_plot_url='/static/auc_plot/task_id_' + str(train_task_id) + "/auc.png",
    )
    # 训练完成修改状态 标记为线性模型
    train_task.model_if_linear = True
    train_task.task_status = 1
    train_task.save()
    return "Scorecard Created"

@celery_app.task
def generate_data_set(generate_data_set_task_id, binning_task_id, data_set_id, mono):
    generate_data_set_task = models.generate_data_set_task.objects.get(id = generate_data_set_task_id)
    binning_task = models.binning_task.objects.get(id = binning_task_id)
    data_set = models.data_set.objects.get(id = data_set_id)
    bins = joblib.load(f"{binning_task.pkl_file_url}binning_task_{binning_task.pk}.pkl")
    data_set_url = data_set.data_url
    data_set_label = data_set.data_label
    original_data = pd.read_csv(data_set_url)
    special_code_list = None
    if data_set.special_code != "":
        special_code_list = data_set.special_code.split(',')

    generated_data = None

    if generate_data_set_task.generate_type == "one-hot":
        feature_splits = {}
        features = []
        for f in bins:
            features.append(f)
            temp = list(bins[f].breaks.values)
            # 把breaks_list里面的数字转化为int/float
            for i in range(len(temp)):
                if temp[i] != 'inf':
                    if float(temp[i]) == int(float(temp[i])):
                        temp[i] = int(float(temp[i]))
                    else:
                        temp[i] = float(temp[i])
            if special_code_list is not None:
                for s in special_code_list:  # 分箱列表中移除special值
                    if int(s) in temp:
                        temp.remove(int(s))
            temp.remove('inf')
            print(temp)
            feature_splits[f] = np.array(temp)

        generated_data = generate_data_set_tool.one_side_interval(original_data, features, feature_splits, mono, special_code_list)
        # 加上target列
        generated_data = pd.concat([original_data[data_set_label], generated_data], axis=1, ignore_index=False)
    elif generate_data_set_task.generate_type == "woe": # 生成WoE值替换的数据集
        pass

    if isinstance(generated_data, pd.DataFrame):
        generate_data_set_task.generate_status = 1
        generate_data_set_task.url = f'./generated_data/{generate_data_set_task.generate_data_set_name}.csv'
        generated_data.to_csv(f'./generated_data/{generate_data_set_task.generate_data_set_name}.csv', index=False)
        generate_data_set_task.save()
        return "Dataset Generated Success"
    else:
        generate_data_set_task.generate_status = 0
        generate_data_set_task.save()
        return "Dataset Generated Failed"


@celery_app.task
def train_non_negative_logistic_regression(task_id, label, data_set_url, train_ratio, random_state):
    data = pd.read_csv(data_set_url)
    X_train, X_test, y_train, y_test = train_test_split(data.drop([label], axis=1), data[label], test_size=(100-int(train_ratio)) / 100, random_state=int(random_state))
    X_y_train = pd.concat([y_train, X_train], axis=1, ignore_index=False)
    X_y_test = pd.concat([y_test, X_test], axis=1, ignore_index=False)

    pipeline = Pipeline([
        LogisticRegressionBinaryClassifier(use_threads=False, normalize='No',label=label, enforce_non_negativity=True, l1_regularization=1, l2_regularization=1)
    ])
    model = pipeline.fit(X_y_train)
    # 将model保存为pkl文件
    if not os.path.exists(os.path.join(str(BASE_DIR) + '\\task_results\\task_id_' + str(task_id))):
        os.makedirs(os.path.join(str(BASE_DIR) + '\\task_results\\task_id_' + str(task_id)))
    joblib.dump(model, os.path.join(str(BASE_DIR) + '\\task_results\\task_id_' + str(task_id)) + '\\task_id_' + str(task_id) + ".pkl")

    metrics, predictions = model.test(X_y_test, output_scores=True)

    y_pred = predictions.PredictedLabel.values
    y_pred_proba = predictions.Probability.values

    accuracy, precision, recall, f1_score = model_measure.getPerformance(task_id, "Non_Negative_LogReg_TaskID_"+str(task_id), np.array(y_test), np.array(y_pred), np.array(y_pred_proba))

    # 拼装参数字典
    train_result = {}
    feature_name_list = list(data.columns)
    feature_name_list.remove(label)
    count = 0
    for f in feature_name_list:
        train_result[f] = model.summary()['Weights.' + f].values[0]
        count = count + 1
    train_result['intercept'] = model.summary().Bias.values[0]

    # 保存训练结果信息
    models.linear_task_result.objects.create(
        task_id=models.task.objects.get(id=task_id),
        task_result_str=str(train_result),
        task_result_url="None",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        auc_plot_url='/static/auc_plot/task_id_' + str(task_id) + "/auc.png"
    )

    # 更新任务记录信息
    cur_task = models.task.objects.get(id=task_id)
    cur_task.task_end = timezone.now()
    cur_task.task_status = 1  # 1:训练成功
    cur_task.model_if_linear = True # 是线性模型
    cur_task.save()

    return "Non-negative Logistic Regression Success"