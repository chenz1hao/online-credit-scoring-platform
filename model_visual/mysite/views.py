import json
import os
import ast
import sys
from datetime import datetime

import joblib
from django.http import HttpResponse, FileResponse, QueryDict
from django.shortcuts import render
from django.utils.encoding import escape_uri_path
from django.views import View

from model_visual.settings import BASE_DIR
from . import models
import pandas as pd
import numpy as np
from celery_tasks.tasks import train_process, data_set_bin, update_dataset_bin, create_score_card, generate_data_set, \
    train_non_negative_logistic_regression
import scorecardpy as sc


# 数据集展示列表
class DataList(View):
    # 返回数据集列表视图
    def get(self, request):
        context = {
            'title': '数据列表',
            'dataList': models.data_set.objects.all(),
            'active': 2,
        }
        return render(request, 'data.html', context)

    # 下载某个数据集
    def post(self, request):
        id = request.POST.get('id')
        data = models.data_set.objects.get(id=id)

        file = open(data.data_url, 'rb')
        response = FileResponse(file)
        response['Content-Type'] = 'application/octet-stream'
        response['Content-Disposition'] = 'attachment;filename="' + data.data_file_name + '"'
        return response

    # 删除某个数据集
    def delete(self, request):
        DELETE = QueryDict(request.body)
        id = DELETE.get('id')
        del_obj = models.data_set.objects.get(id=id)
        del_obj_url = del_obj.data_url

        if os.path.exists(del_obj_url):
            #删除文件，可使用以下两种方法。
            os.remove(del_obj_url)

        del_obj.delete()
        return HttpResponse("删除成功")


# 数据集上传页面
class UploadData(View):
    def get(self, request):
        context = {
            'title': '上传数据集',
            'active': 2,
        }
        return render(request, 'upload_data.html', context)

    def post(self, request):
        data_url = request.POST.get('data', None)
        data_desc = request.POST.get('data_desc', None)
        data_name = request.POST.get('data_name', None)
        label_select = request.POST.get('label_select', None)
        special_code = request.POST.get('special_code', None)
        if (not data_url == None and not data_desc == None and not data_name == None and not label_select == None):
            """
             插入data_set表
            """
            startIdx = -1
            count = 0
            for i in data_url[::-1]:
                count = count + 1
                if i == "\\":
                    startIdx = len(data_url) - count + 1
                    break

            data_file_name = data_url[startIdx:]
            data_url = './data/' + data_file_name
            df = pd.read_csv(data_url)
            data_feature_nums = df.shape[1] - 1
            data_sample_nums = df.shape[0]
            m = models.data_set.objects.create(
                data_name=data_name,
                data_upload_date=datetime.now(),
                data_desc=data_desc,
                data_feature_nums=data_feature_nums,
                data_sample_nums=data_sample_nums,
                data_url=data_url,
                data_file_size=self.get_FileSize(data_url),
                data_file_name=data_file_name,
                data_label=label_select,
                special_code=special_code,
            )
            # 异步开始
            """
             Celery同步更新data_set_bin分箱表
            """
            # 分箱的工作不放在这里
            # data_set_bin_and_one_hot.delay(m.id, data_url, data_file_name)
            # 异步结束
            return HttpResponse("上传成功")
        else:
            return HttpResponse("上传失败")

    def get_FileSize(self, filePath):
        fsize = os.path.getsize(filePath)
        fsize = fsize / float(1024 * 1024)
        return round(fsize, 2)


# 选择数据集后，因为要预览数据，所以需要单独把一个数据文件提交到服务器存储后再返回回去。【注：该类不涉及与数据库的交互】
class UploadTempData(View):
    def post(self, request):
        data = request.FILES.get('data')
        if not data == None:
            file_name = str(data)

            # 判断数据库中是否已经存在该名字的书籍
            check_exist = models.data_set.objects.filter(data_file_name=file_name)
            if check_exist.exists():
                return HttpResponse("exist")

            with open('data/' + file_name, 'wb') as file_obj:
                for chunk in data.chunks():
                    file_obj.write(chunk)

            df = pd.read_csv('data/' + file_name).reset_index(drop=True)

            json_dict = json.loads(df.to_json(orient='table'))
            return HttpResponse(json.dumps(json_dict))
        return HttpResponse(None)


# 任务列表
class TaskList(View):
    def get(self, request):
        context = {
            'title': '模型训练',
            'active': 1,
            'tasks': models.task.objects.all().order_by("-task_start"),
        }
        return render(request, 'task.html', context)

    def delete(self, request):
        task_id = QueryDict(request.body).get('task_id')
        task_obj = models.task.objects.get(id=task_id)
        task_obj.delete()
        return HttpResponse("删除成功")


# 开始一个新的训练任务
class StartTask(View):
    def get(self, request):
        context = {
            'title': '开始一个任务',
            'active': 1,
            'models': models.model_class.objects.exclude(model_name='score_card').exclude(
                model_name='non_negative_LogReg'),
            'dataset': models.data_set.objects.all(),
        }
        return render(request, 'start_task.html', context)

    def post(self, request):
        model_select = request.POST.get('model_select', None)
        data_select = request.POST.get('data_select', None)
        train_test_split = request.POST.get('train_test_split', None)
        random_state = request.POST.get('random_state', None)

        t = models.task.objects.create(
            model_id=models.model_class.objects.get(id=model_select),
            task_start=datetime.now(),
            task_end=datetime.now(),
            task_status=0,
            data_set_id=models.data_set.objects.get(id=data_select),
            model_run_train_test_split=train_test_split,
            model_run_random_state=random_state,
            related_binning_task=-1,
        )
        # 查询该数据集的label列名，也就是y
        label_name = models.data_set.objects.get(id=data_select).data_label
        # Celery异步执行模型训练任务开始
        train_process.delay(t.id, model_select, data_select, train_test_split, random_state, label_name)
        # 异步结束

        return HttpResponse("任务创建成功")


# 任务对比
class TaskCompare(View):
    def post(self, request):
        tasks_str = request.POST.get('tasks', None)
        tasks_str = tasks_str[1:len(tasks_str) - 1]
        tasks_id = tasks_str.split(",")
        tasks = []
        for i in tasks_id:
            task_obj = models.linear_task_result.objects.get(task_id=int(i))
            task = {
                'task_id': task_obj.task_id.id,
                'accuracy': task_obj.accuracy,
                'precision': task_obj.precision,
                'recall': task_obj.recall,
                'f1_score': task_obj.f1_score,
                'auc_plot_url': task_obj.auc_plot_url
            }
            tasks.append(task)

        return HttpResponse(json.dumps(tasks))


# 分箱详细信息
class DatasetBin(View):
    def get(self, request, id):
        data_set_bins = list(models.data_set_bin.objects.filter(data_set_id=id))
        var_split_dict = {}
        for bin in data_set_bins:
            bin_list = eval(bin.bin_result)
            bin_list_int = []
            for section in bin_list:
                section = section[1:-1]
                split_idx = section.index(",")
                first_num, second_num = section[:split_idx], section[split_idx + 1:]
                if first_num == "-INF":
                    first_num = -999
                if second_num == "+INF":
                    second_num = 999
                bin_list_int.append([float(first_num), float(second_num)])

            var_split_dict[bin.feature_name] = bin_list_int
        print(var_split_dict)
        context = {
            'title': '数据集ID' + str(id) + " 分箱信息",
            'active': 2,
            'var_split_dict': var_split_dict,
        }
        return render(request, "bin.html", context)


# 线性模型查看系数
class TaskResult(View):
    def get(self, request, id):
        # 判断当前选择模型是否是线性模型
        cur_task = models.task.objects.get(id=id)
        if cur_task.model_if_linear == True:
            task_result = models.linear_task_result.objects.get(task_id=id)
            task_result_dict = ast.literal_eval(task_result.task_result_str)
            accuracy = task_result.accuracy
            precision = task_result.precision
            recall = task_result.recall
            f1_score = task_result.f1_score
            context = {
                'title': '任务' + str(id) + '的系数结果',
                'active': 1,
                'task_result_dict': task_result_dict,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
            }
            return render(request, "task_result.html", context)
        else:
            context = {
                'title': '任务' + str(id) + '的系数结果',
                'tips': '非线性模型或训练未成功的任务无法查看系数'
            }
            return render(request, "task_result.html", context)


# 分箱任务列表
class BinningLibrary(View):
    def get(self, request):
        context = {
            'title': '分箱任务列表',
            'active': 3,
            'tasks': models.binning_task.objects.all(),
        }
        return render(request, "binning_library.html", context)

    def delete(self, request):
        del_id = QueryDict(request.body).get('task_id')
        del_obj = models.binning_task.objects.get(id=del_id)
        # 删除分箱结果信息
        models.binning_feature.objects.filter(binning_task_id=del_obj).delete()
        # 删除生成数据任务记录
        del_generate_dataset_tasks = models.generate_data_set_task.objects.filter(binning_task=del_obj)

        for task in del_generate_dataset_tasks:
            if os.path.exists(os.path.join(str(BASE_DIR) + f'./generated_data/{task.generate_data_set_name}.csv')):
                os.remove(os.path.join(str(BASE_DIR) + f'./generated_data/{task.generate_data_set_name}.csv'))

        del_generate_dataset_tasks.delete()
        del_obj.delete()

        return HttpResponse("删除成功")


class StartBinning(View):
    def post(self, request):  # 开始分箱任务
        binning_task_name = request.POST.get('binning_task_name', None)
        data_id = request.POST.get('data_id', None)
        data_set = models.data_set.objects.get(id=data_id)

        label_select = request.POST.get('label_select', None)
        train_ratio = request.POST.get('train_ratio', None)
        random_state = request.POST.get('random_state', None)
        binning_feature_list = request.POST.get('binning_feature_list', None).split(",")
        special_code = data_set.special_code

        t = models.binning_task.objects.create(
            binning_task_name=binning_task_name,
            data_set_id=data_set,
            task_status=0,
            task_start=datetime.now(),
            training_label=label_select,
            train_ratio=train_ratio,
            random_state=random_state,
        )
        # 创建分箱异步任务
        data_set_bin.delay(t.id, data_id, label_select, train_ratio, random_state, binning_feature_list, special_code)

        return HttpResponse('任务创建成功')

    def get(self, request):
        context = {
            'title': '开始分箱任务',
            'dataset': models.data_set.objects.all(),
            'active': 3,
        }
        return render(request, "start_binning.html", context)


class GetDatasetFeatures(View):
    def post(self, request):
        data_id = request.POST.get('data_id', None)
        # 查找这个数据的存储位置
        data_url = models.data_set.objects.get(id=data_id).data_url
        features = list(pd.read_csv(data_url).reset_index(drop=True).columns.values)
        feature_dict = {}
        feature_dict['features'] = features
        return HttpResponse(json.dumps(feature_dict))


class BinningResult(View):
    def get(self, request, id):
        binning_task = models.binning_task.objects.get(id=id)
        feature_name = models.binning_feature.objects.filter(binning_task_id=binning_task).values('feature_name')
        unique_feature_name = []
        binning_dict = {}

        # 剔除出重复的feature_name
        for f in feature_name:
            if len(unique_feature_name) == 0:
                unique_feature_name.append(f['feature_name'])
            else:
                if not f['feature_name'] == unique_feature_name[-1]:
                    unique_feature_name.append(f['feature_name'])

        for f in unique_feature_name:
            value = []
            cur_feature_binning = models.binning_feature.objects.filter(binning_task_id=binning_task, feature_name=f)

            for row in cur_feature_binning:
                this_bin_dict = {}
                this_bin_dict['bin_num'] = row.bin_num
                this_bin_dict['which_bin'] = row.which_bin
                this_bin_dict['this_bin_str'] = row.this_bin_str
                this_bin_dict['count'] = row.count
                this_bin_dict['count_percent'] = row.count_percent
                this_bin_dict['non_event_num'] = row.non_event_num
                this_bin_dict['event_num'] = row.event_num
                this_bin_dict['event_rate'] = row.event_rate
                this_bin_dict['woe'] = row.woe
                this_bin_dict['this_iv'] = row.this_iv
                this_bin_dict['total_iv'] = row.total_iv
                this_bin_dict['is_special_values'] = str(row.is_special_values)
                this_bin_dict['breaks'] = row.breaks

                value.append(this_bin_dict)

            binning_dict[f] = value

        context = {
            'title': '分箱任务ID' + str(id) + '结果',
            'active': 3,
            'binning_task_id': str(id),
            'feature_name': unique_feature_name,
            'binning_dict': binning_dict,
        }
        return render(request, "binning_result.html", context)


class BinningUpdate(View):
    def get(self, request, id):
        binning_task = models.binning_task.objects.get(id=id)
        # bins = models.binning_feature.objects.filter(binning_task_id = binning_task)
        feature_name = models.binning_feature.objects.filter(binning_task_id=binning_task).values('feature_name')
        unique_feature_name = []
        features_breaks = {}
        # special_code = binning_task.data_set_id.special_code.split(",")

        # 剔除出重复的feature_name
        for f in feature_name:
            if len(unique_feature_name) == 0:
                unique_feature_name.append(f['feature_name'])
            else:
                if not f['feature_name'] == unique_feature_name[-1]:
                    unique_feature_name.append(f['feature_name'])

        # 找出每个特征的分割点用于用户编辑
        for f in unique_feature_name:
            this_feature_breaks = models.binning_feature.objects.filter(binning_task_id=binning_task, feature_name=f,
                                                                        is_special_values=False).values('breaks')
            features_breaks[f] = this_feature_breaks

        context = {
            'title': '分箱任务ID' + str(id) + '结果',
            'active': 3,
            'binning_task_id': str(id),
            'feature_name': unique_feature_name,
            'features_breaks': features_breaks,
        }

        return render(request, "binning_update.html", context)

    def post(self, request, id):
        binning_task = models.binning_task.objects.get(id=id)
        feature_name = models.binning_feature.objects.filter(binning_task_id=binning_task).values('feature_name')
        unique_feature_name = []
        binning_dict = {}
        special_code = binning_task.data_set_id.special_code.split(",")

        # 剔除出重复的feature_name
        for f in feature_name:
            if len(unique_feature_name) == 0:
                unique_feature_name.append(f['feature_name'])
            else:
                if not f['feature_name'] == unique_feature_name[-1]:
                    unique_feature_name.append(f['feature_name'])

        # 将binning
        index = 1
        for f in unique_feature_name:
            this_feature_bin = []
            while not request.POST.get(f + "_" + str(index), None) == None:
                param = request.POST.get(f + "_" + str(index), None)
                if param == "inf":
                    index += 1
                    continue
                this_feature_bin.append(param)
                index += 1
            binning_dict[f] = this_feature_bin
            index = 1

        # 异步更新分箱
        binning_task.task_status = 3  # 3表示分箱更新中
        binning_task.save()
        update_dataset_bin.delay(id, binning_dict, special_code, unique_feature_name)

        return HttpResponse("分箱更新中")


# 创建评分卡任务类
class CreateScoreCard(View):
    def get(self, request):
        context = {
            'title': '创建评分卡',
            'active': 1,
            'binning_library': models.binning_task.objects.filter(task_status=1),
        }
        return render(request, 'create_score_card.html', context)

    def post(self, request):
        binning_task_id = request.POST.get('binning_select')
        binning_task = models.binning_task.objects.get(id=binning_task_id)
        scorecard_feature_list = request.POST.get('scorecard_feature_list').split(',')

        t = models.task.objects.create(
            model_id=models.model_class.objects.get(model_name="score_card"),
            task_start=datetime.now(),
            task_end=datetime.now(),
            task_status=0,
            data_set_id=binning_task.data_set_id,
            model_run_train_test_split=binning_task.train_ratio,
            model_run_random_state=binning_task.random_state,
            related_binning_task=binning_task_id,
        )

        create_score_card.delay(t.id, binning_task_id, scorecard_feature_list)
        return HttpResponse("评分卡创建成功")


class GetBinnedFeatures(View):
    def post(self, request):
        binning_task_id = request.POST.get('binning_select', None)
        binning_task = models.binning_task.objects.get(id=binning_task_id)
        feature_name = models.binning_feature.objects.filter(binning_task_id=binning_task).values('feature_name')
        unique_feature_name = []

        # 剔除出重复的feature_name
        for f in feature_name:
            if len(unique_feature_name) == 0:
                unique_feature_name.append(f['feature_name'])
            else:
                if not f['feature_name'] == unique_feature_name[-1]:
                    unique_feature_name.append(f['feature_name'])

        return HttpResponse(json.dumps(unique_feature_name))


class ScoreCard(View):
    def get(self, request, id):
        # 读取评分卡pkl
        score_card = joblib.load(f'./score_card/score_card_id_{id}.pkl')
        score_card_dict_list = {}
        for key in score_card:
            this_feature_list = []
            for i in range(len(score_card[key])):
                this_feature_this_bin_list = []
                this_feature_this_bin_list.append(score_card[key].iloc[i]['variable'])
                this_feature_this_bin_list.append(score_card[key].iloc[i]['bin'])
                this_feature_this_bin_list.append(score_card[key].iloc[i]['points'])
                this_feature_list.append(this_feature_this_bin_list)
            score_card_dict_list[key] = this_feature_list

        context = {
            'title': f'训练任务ID{id}_评分卡展示',
            'active': 1,
            'score_card': score_card_dict_list,
        }
        return render(request, "score_card.html", context)


class GeneratedDatasetList(View):
    def get(self, request):
        generate_data_set_task = models.generate_data_set_task.objects.all().order_by('-id')
        context = {
            'title': '衍生数据列表',
            'active': 4,
            'generate_data_set_task': generate_data_set_task,
        }
        return render(request, "generated_dataset_list.html", context)

    def post(self, request):
        generate_task = models.generate_data_set_task.objects.get(id=request.POST.get('id'))

        file = open(generate_task.url, 'rb')
        response = FileResponse(file)
        response['Content-Type'] = 'application/octet-stream'
        # response['Content-Disposition'] = 'attachment;filename="' + generate_task.generate_data_set_name + '.csv"'
        response['Content-Disposition'] = "attachment; filename*=utf-8''{}".format(
            escape_uri_path(generate_task.generate_data_set_name + ".csv"))
        return response

    def delete(self, request):
        del_id = QueryDict(request.body).get('id')
        del_obj = models.generate_data_set_task.objects.get(id=del_id)
        # 删除生成的数据文件
        if os.path.exists(os.path.join(str(BASE_DIR) + f'./generated_data/{del_obj.generate_data_set_name}.csv')):
            os.remove(os.path.join(str(BASE_DIR) + f'./generated_data/{del_obj.generate_data_set_name}.csv'))
        # 删除生成数据任务记录
        del_obj.delete()
        return HttpResponse("删除成功")


class StartGenerateData(View):
    def get(self, request):
        binning_task = models.binning_task.objects.filter(task_status=1)
        context = {
            'title': '衍生数据',
            'active': 4,
            'binning_task': binning_task,
        }
        return render(request, "start_generate_dataset.html", context)

    def post(self, request):
        generate_data_set_name = request.POST.get('generate_data_set_name')
        binning_task_id = request.POST.get('binning_task_id')
        generate_type = request.POST.get('generate_type')
        data_set_id = models.binning_task.objects.get(id=binning_task_id).data_set_id_id
        mono_setting = request.POST.get('mono', None)

        g = models.generate_data_set_task.objects.create(
            generate_data_set_name=generate_data_set_name,
            binning_task=models.binning_task.objects.get(id=binning_task_id),
            data_set=models.data_set.objects.get(id=data_set_id),
            generate_type=generate_type,
            generate_status=0,
        )
        if mono_setting != None:
            mono_setting_dict = json.loads(mono_setting)
            g.mono = mono_setting
            g.save()

        generate_data_set.delay(g.id, binning_task_id, data_set_id, mono_setting_dict)

        return HttpResponse("数据集衍生任务创建成功")


class StartNonNegativeLogRegSC(View):
    def get(self, request):
        context = {
            'title': '非负逻辑回归在one_hot数据集上创建评分卡',
            'active': 1,
            'one_hot_dataset': models.generate_data_set_task.objects.filter(generate_status=1)
        }
        return render(request, "start_non_negative_log_reg_score_card.html", context)

    def post(self, request):
        one_hot_dataset_id = request.POST.get('data_select')
        generated_dataset = models.generate_data_set_task.objects.get(id=one_hot_dataset_id)
        # 这个url应该是衍生数据集的URL 不能是原数据集的URL
        data_set_url = models.generate_data_set_task.objects.get(id=one_hot_dataset_id).url
        train_ratio = request.POST.get('train_ratio')
        random_state = request.POST.get('random_state')
        t = models.task.objects.create(
            model_id=models.model_class.objects.get(model_name='non_negative_LogReg'),
            task_status=0,
            data_set_id=generated_dataset.data_set,
            task_start=datetime.now(),
            task_end=datetime.now(),
            model_run_train_test_split=train_ratio,
            model_run_random_state=random_state,
            generated_dataset_id=one_hot_dataset_id,
            related_binning_task=-1,
        )
        train_non_negative_logistic_regression.delay(t.id, generated_dataset.data_set.data_label, data_set_url,
                                                     train_ratio, random_state)

        return HttpResponse("任务创建成功")


# 这个view用于非负系数逻辑回归的结果转评分卡形式
class CustomScoreCard(View):
    def get(self, request, id):
        task_obj = models.task.objects.get(pk=id)
        generated_dataset_id = task_obj.generated_dataset_id
        if generated_dataset_id == -1:
            return HttpResponse("只有非负逻辑回归任务可以生成此类评分卡")

        generate_data_set_task = models.generate_data_set_task.objects.get(id=generated_dataset_id)
        binning_task = generate_data_set_task.binning_task

        feature_name = models.binning_feature.objects.filter(binning_task_id=binning_task).values('feature_name')
        unique_feature_name = []

        # 剔除出重复的feature_name
        for f in feature_name:
            if len(unique_feature_name) == 0:
                unique_feature_name.append(f['feature_name'])
            else:
                if not f['feature_name'] == unique_feature_name[-1]:
                    unique_feature_name.append(f['feature_name'])

        # 取出分箱的结果
        binning_res = {}
        binning_feature = models.binning_feature.objects.filter(binning_task_id=binning_task)
        for f in unique_feature_name:
            binning_res[f] = []
            this_binning_feature = binning_feature.filter(feature_name=f)
            for t in this_binning_feature:
                binning_res[f].append(t.this_bin_str)

        # 找出衍生数据ID 获得特征单调性
        generated_data_set_task = models.generate_data_set_task.objects.get(pk=task_obj.generated_dataset_id)
        mono = generated_data_set_task.mono
        mono_dict = eval(mono)

        # 获取该原始数据集的special_code
        original_data_set = generated_data_set_task.data_set
        special_code_list = None
        if original_data_set.special_code != "":
            special_code_list = original_data_set.special_code.split(',')

        # 开始拼装这个score_card
        score_card_dict_list = {}
        task_result_str = models.linear_task_result.objects.get(task_id=task_obj).task_result_str
        task_result_dict = eval(task_result_str)

        feature_generated_dict = {}
        for f in unique_feature_name:
            feature_generated_dict[f] = []
            this_feature_mono = mono_dict.get(f, None)
            score_card_dict_list[f] = []
            # 这个特征没有单调性约束
            if this_feature_mono == None or this_feature_mono == 0:
                for key in task_result_dict:
                    # 这里注意，如果特征v1是需要的，特征v10 v11也会进入，所以要加两个条件判断
                    if (f in key and f+"_" in key) or f+" " in key:
                        this_interval = []
                        this_interval.append(f)
                        # 通过get_this_bin_str_no_mono()方法来获取this_bin_str
                        this_bin_str = self.get_this_bin_str(key, special_code_list)
                        this_interval.append(this_bin_str)
                        this_interval.append(task_result_dict[key])
                        score_card_dict_list[f].append(this_interval)
            # 有单调性约束的
            elif this_feature_mono in [1, 2]:
                for key in task_result_dict:
                    if (f in key and f+"_" in key) or f+" " in key:
                        feature_generated_dict[f].append(key)
                prior_generated_name_idx = -1
                after_generated_name_idx = 1
                for key in task_result_dict:
                    if (f in key and f+"_" in key) or f+" " in key:
                        this_interval = []
                        this_interval.append(f)
                        # 通过get_this_bin_str_no_mono()方法来获取this_bin_str
                        if this_feature_mono == 1:
                            if prior_generated_name_idx == -1:
                                this_bin_str = self.get_this_bin_str_with_mono(None, key, special_code_list,
                                                                               this_feature_mono)
                            else:
                                this_bin_str = self.get_this_bin_str_with_mono(
                                    feature_generated_dict[f][prior_generated_name_idx], key, special_code_list,
                                    this_feature_mono)
                            prior_generated_name_idx += 1
                        elif this_feature_mono == 2:
                            if after_generated_name_idx + 1 > len(feature_generated_dict[f]):
                                this_bin_str = self.get_this_bin_str_with_mono(key, None, special_code_list,
                                                                               this_feature_mono)
                            else:
                                this_bin_str = self.get_this_bin_str_with_mono(key, feature_generated_dict[f][
                                    after_generated_name_idx], special_code_list, this_feature_mono)
                            after_generated_name_idx += 1
                        this_interval.append(this_bin_str)
                        this_interval.append(task_result_dict[key])
                        score_card_dict_list[f].append(this_interval)

                # 对单调约束的特征进行分数累加
                if this_feature_mono == 1:  # 减
                    cum_point = 0
                    for i in range(len(score_card_dict_list[f]) - 1, -1, -1):
                        if special_code_list != None and self.if_special_code_in_feature_name(special_code_list,
                                                                                              score_card_dict_list[f][i][1]):  # score_card_dict_list[f][i][1] in special_code_list:
                            continue
                        this_item = score_card_dict_list[f][i][2]
                        score_card_dict_list[f][i][2] += cum_point
                        cum_point += this_item
                elif this_feature_mono == 2:  # 增
                    cum_point = 0
                    for i in range(len(score_card_dict_list[f])):
                        if special_code_list != None and self.if_special_code_in_feature_name(special_code_list,
                                                                                              score_card_dict_list[f][i][1]):
                            continue
                        this_item = score_card_dict_list[f][i][2]
                        score_card_dict_list[f][i][2] += cum_point
                        cum_point += this_item

        for f in unique_feature_name:
            this_feature_mono = mono_dict.get(f, None)
            mono_txt = ""
            if this_feature_mono == 0:
                mono_txt = "无单调约束"
            elif this_feature_mono == 1:
                mono_txt = "单减约束"
            elif this_feature_mono == 2:
                mono_txt = "单增约束"

            score_card_dict_list[f + " " + mono_txt] = score_card_dict_list.pop(f)


        context = {
            'title': f'训练任务ID{id}_评分卡展示',
            'active': 1,
            'score_card': score_card_dict_list,
        }
        return render(request, "score_card.html", context)

    def get_this_bin_str(self, generated_feature_name, special_code_list):
        res = ""
        # 如果只有 < 没有 <= 第一个区间 应该返回[-inf,xxx)格式
        if '<' in generated_feature_name and '<=' not in generated_feature_name:
            res += "[-inf,"
            res += generated_feature_name[generated_feature_name.find('<') + 1:]
            res += ")"
        # 既有 < 又有 <= 中间区间 返回[x1, x2)格式
        elif '<' in generated_feature_name and '<=' in generated_feature_name:
            res += "["
            first_num = generated_feature_name[generated_feature_name.find(' ') + 1: generated_feature_name.find('<=')]
            res += first_num + ","
            second_num = generated_feature_name[generated_feature_name.find('X<') + 2:]
            res += second_num + ")"

        # 只有 >= 最后区间 返回[xxx, +inf)格式
        elif '>=' in generated_feature_name:
            res += "["
            res += generated_feature_name[generated_feature_name.find('>=') + 2:]
            res += ",+inf)"
        # 特殊值区间 直接返回特殊值即可
        elif special_code_list != None and self.if_special_code_in_feature_name(special_code_list,
                                                                                generated_feature_name):
            res = generated_feature_name[generated_feature_name.find('_') + 1:]
        return res

    def get_this_bin_str_with_mono(self, prior_generated_feature_name, generated_feature_name, special_code_list, mono):
        res = ""
        if mono == 1:
            # 先判断是不是特殊值
            if special_code_list != None and self.if_special_code_in_feature_name(special_code_list,
                                                                                  generated_feature_name):
                res += generated_feature_name[generated_feature_name.find('_') + 1:]
            elif prior_generated_feature_name == None:
                num = generated_feature_name[generated_feature_name.find('<') + 1:]
                res += ("[-inf," + num + ")")
            else:
                res += "["
                res += prior_generated_feature_name[prior_generated_feature_name.find('<') + 1:]
                res += ","
                res += generated_feature_name[generated_feature_name.find('<') + 1:]
                res += ")"
        elif mono == 2:
            # 先判断是不是特殊值
            if special_code_list != None and self.if_special_code_in_feature_name(special_code_list,
                                                                                  prior_generated_feature_name):
                res += prior_generated_feature_name[prior_generated_feature_name.find('_') + 1:]
            elif "-INF" in prior_generated_feature_name:  # 第一条
                res += "[-inf,"
                res += generated_feature_name[generated_feature_name.find('>=') + 2:]
                res += ")"
            elif special_code_list != None and self.if_special_code_in_feature_name(special_code_list,
                                                                                    generated_feature_name):
                res += "["
                res += prior_generated_feature_name[prior_generated_feature_name.find('>=') + 2:]
                res += ",+inf)"
            elif special_code_list == None and generated_feature_name == None:
                res += "["
                res += prior_generated_feature_name[prior_generated_feature_name.find('>=') + 2:]
                res += ",+inf)"
            else:
                res += "["
                res += prior_generated_feature_name[prior_generated_feature_name.find('>=') + 2:]
                res += ","
                res += generated_feature_name[generated_feature_name.find('>=') + 2:]
                res += ")"

        return res

    """
    判断generated_feature_name是否是属于特殊分箱的区间
    如果属于特殊分箱则返回True，否则返回False
    """

    def if_special_code_in_feature_name(self, special_code_list, generated_feature_name):
        for s in special_code_list:
            if s in generated_feature_name:
                return True
        return False


"""
下一步对单个样本提供预测 提供解释
"""


class SamplePredict(View):
    def get(self, request, id):
        task_obj = models.task.objects.get(id=id)
        model_name = task_obj.model_id.model_name
        data_obj = task_obj.data_set_id
        data_url = data_obj.data_url
        if model_name in ["score_card"]:
            model = joblib.load(f'score_card/score_card_id_{id}.pkl')
            features = list(model.keys())
            features.remove('basepoints')
        else:
            label = data_obj.data_label
            data = pd.read_csv(data_url)
            cols = data.columns.values
            features = list(cols)
            features.remove(label)

        context = {
            'title': f'训练任务ID{id}_单样本预测',
            'active': 1,
            'features': features,
            'task_id': id
        }
        return render(request, "sample_predict.html", context)

    def post(self, request):

        task_id = request.POST.get("task_id")
        task_obj = models.task.objects.get(id=task_id)
        model_name = task_obj.model_id.model_name
        result = {}

        """
        两种训练任务涉及到单个样本的汉明距离解释 ：score_card non_negative_LogReg
        但是这两种要分别处理，因为这两种模型的模型类不是完全一样的，不能统一处理
        """
        if model_name == "score_card":
            sample_need_predict = []
            # 导入之前训练的pkl文件
            card = joblib.load(f'score_card/score_card_id_{task_id}.pkl')
            lr = joblib.load(f'task_results/task_id_{task_id}/task_id_{task_id}.pkl')
            binning_task_obj = models.binning_task.objects.get(id = task_obj.related_binning_task)#models.generate_data_set_task.objects.filter(binning_task_id=task_obj.related_binning_task)[0].binning_task
            binning_task_dict = joblib.load(
                binning_task_obj.pkl_file_url + 'binning_task_' + str(task_obj.related_binning_task) + ".pkl")

            # 一定要按照训练时候特征的排列顺序来，并组装为pd.DataFrame的格式（woebin_ply方法要用到）
            features = list(card.keys())
            features.remove('basepoints')
            for f in features:
                if "." in request.POST.get(f):
                    sample_need_predict.append(float(request.POST.get(f)))
                else:
                    sample_need_predict.append(int(request.POST.get(f)))
            test_sample = pd.DataFrame([sample_need_predict], columns=features)
            # 将原始特征转化为WoE，但转为WoE后，顺序可能不一样了，要调换特征顺序
            sample_woe = sc.woebin_ply(test_sample, binning_task_dict)
            order = [f + "_woe" for f in features]
            sample_woe = sample_woe[order]

            # 导入逻辑回归PKL进行预测
            predict = lr.predict(sample_woe)
            result['predict'] = int(predict)

            # 找出汉明距离最近的top5个样本
            all_samples = pd.read_csv(task_obj.data_set_id.data_url)[features]
            nearest_samples = self.getTopKHanmingNearestSamples(test_sample, all_samples, binning_task_dict, 5)
            i = 0
            for sample in nearest_samples:
                this_sample_dict = {}
                for col_name in all_samples.iloc[sample[1]].keys():
                    this_sample_dict[col_name] = all_samples.iloc[sample[1]][col_name]
                nearest_samples[i].append(this_sample_dict)
                i += 1
            result['nearest_samples'] = nearest_samples


        result = str(result)
        result = eval(result)
        return HttpResponse(json.dumps(result))

    """
    返回所有样本中距离test_sample one-hot汉明距离最近的K个样本下标
    """
    def getTopKHanmingNearestSamples(self, test_sample, all_samples, bins_dict, k):
        nearest_k_samples = []
        # 根据bins_dict来转化one_hot
        test_sample_one_hot = self.trans_into_one_hot(test_sample.iloc[0], bins_dict)
        all_samples_one_hot = []
        for index, row in all_samples.iterrows():
            all_samples_one_hot.append(self.trans_into_one_hot(row, bins_dict))

        already_search_list = []
        for i in range(k):
            nearest_sample_dis, nearest_sample_idx = self.nearest_sample(test_sample_one_hot, all_samples_one_hot, already_search_list)
            nearest_k_samples.append([nearest_sample_dis, nearest_sample_idx])
            already_search_list.append(nearest_sample_idx)

        return nearest_k_samples



    def trans_into_one_hot(self, sample, bins_dict):
        res = []
        for f in list(sample.keys()):
            this_value = int(sample[f])
            breaks_list = list(bins_dict[f]['breaks'])
            bin_str_list = list(bins_dict[f]['bin'])
            is_special_values = list(bins_dict[f]['is_special_values'])
            has_filled_special_col = False
            this_feature_res = [0 for i in range(len(bin_str_list))]
            for i in range(len(bin_str_list)):
                # 有可能满足多个special_code，所以这里执行完不能马上break
                if is_special_values[i] == True and this_value == int(float(bin_str_list[i])):
                    this_feature_res[i] = 1
                    has_filled_special_col = True
                # 如果是特殊值就不可能还在正常分箱区间内了
                elif is_special_values[i] == False and has_filled_special_col == False:
                    if '-inf' in bin_str_list[i]:
                        num = float(bin_str_list[i][bin_str_list[i].find(',') + 1: len(bin_str_list[i]) - 1])
                        if this_value < num:
                            this_feature_res[i] = 1
                    elif '-inf' not in bin_str_list[i] and 'inf' in bin_str_list[i]:
                        num = float(bin_str_list[i][1: bin_str_list[i].find(',')])
                        if this_value >= num:
                            this_feature_res[i] = 1
                    else:
                        first_num = float(bin_str_list[i][1: bin_str_list[i].find(',')])
                        second_num = float(bin_str_list[i][bin_str_list[i].find(',') + 1: len(bin_str_list[i]) - 1])
                        if first_num <= this_value < second_num:
                            this_feature_res[i] = 1
            # print(bin_str_list)
            # print(this_feature_res)
            res.extend(this_feature_res)
        return res


    def distance(self, sample_1, sample_2):
        if len(sample_1) != len(sample_2):
            return "长度不同无法比较"
        res = 0
        for i in range(len(sample_1)):
            if sample_1[i] != sample_2[i]:
                res += 1
        return res

    def nearest_sample(self, test_sample, all_samples, already_search_list):
        nearest_dis = sys.maxsize
        nearest_dis_idx = -1
        for i in range(len(all_samples)):
            if i in already_search_list: # 已经找过的就不需要再管了
                continue
            cur_dis = self.distance(test_sample, all_samples[i])
            if nearest_dis > cur_dis:
                nearest_dis = cur_dis
                nearest_dis_idx = i
                if nearest_dis == 0: # 距离为0的话之后不可能有有样本距离比这个还近了 可以直接break 提高效率
                    break

        return nearest_dis, nearest_dis_idx
