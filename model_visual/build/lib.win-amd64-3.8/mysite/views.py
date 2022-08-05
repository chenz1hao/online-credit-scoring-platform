import json
import os
from datetime import datetime
from django.http import HttpResponse, FileResponse, QueryDict
from django.shortcuts import render
from django.views import View

from . import models
import pandas as pd
from celery_tasks.tasks import data_set_bin_and_one_hot, train_process

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
        data = models.data_set.objects.get(id = id)

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

        if(not data_url==None and not data_desc==None and not data_name==None):
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
                data_name = data_name,
                data_upload_date = datetime.now(),
                data_desc = data_desc,
                data_feature_nums = data_feature_nums ,
                data_sample_nums = data_sample_nums,
                data_url = data_url,
                data_file_size = self.get_FileSize(data_url),
                data_file_name = data_file_name
            )
            # 异步开始
            """
             Celery同步更新data_set_bin分箱表
            """
            data_set_bin_and_one_hot.delay(m.id, data_url, data_file_name)
            # 异步结束

            return HttpResponse("上传成功（分箱、ONE-HOT编码任务在后台进行中）")
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
            check_exist = models.data_set.objects.filter(data_file_name = file_name)
            if check_exist.exists():
                return HttpResponse("exist")

            with open('data/'+file_name, 'wb') as file_obj:
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
            'tasks': models.task.objects.all(),
        }
        return render(request, 'task.html', context)


# 开始一个新的训练任务
class StartTask(View):
    def get(self, request):
        context = {
            'title': '开始一个任务',
            'active': 1,
            'models': models.model_class.objects.all(),
            'dataset': models.data_set.objects.all(),
        }
        return render(request, 'start_task.html', context)

    def post(self, request):
        model_select = request.POST.get('model_select', None)
        data_select = request.POST.get('data_select', None)
        train_test_split = request.POST.get('train_test_split', None)
        random_state = request.POST.get('random_state', None)
        if_one_hot = False if request.POST.get('if_one_hot', None)==None else True

        t = models.task.objects.create(
            model_id = models.model_class.objects.get(id=model_select),
            task_start = datetime.now(),
            task_end = datetime.now(),
            task_status = 0,
            data_set_id = models.data_set.objects.get(id=data_select),
            model_run_train_test_split = train_test_split,
            model_run_random_state = random_state,
            if_use_one_hot = if_one_hot
        )

        # Celery异步执行模型训练任务开始
        train_process.delay(t.id, model_select, data_select, train_test_split, random_state, if_one_hot)
        # 异步结束

        return HttpResponse("任务创建成功")

# 任务对比
class TaskCompare(View):
    def post(self, request):
        tasks_str = request.POST.get('tasks', None)
        tasks_str = tasks_str[1:len(tasks_str)-1]
        tasks_id = tasks_str.split(",")
        tasks = []
        for i in tasks_id:
            task_obj = models.linear_task_result.objects.get(task_id = int(i))
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
