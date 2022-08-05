from django.utils import timezone

from django.db import models
"""
数据集表(csv_data):
ID
数据集名
上传时间
数据集描述
特征数
样本数
文件大小
数据URL
"""
class data_set(models.Model):
    data_name = models.CharField(verbose_name="数据集名称", max_length=100)
    data_upload_date = models.DateField(verbose_name="上传时间")
    data_desc = models.CharField(verbose_name="数据集描述", max_length=100)
    data_feature_nums = models.IntegerField(verbose_name="特征数")
    data_sample_nums = models.IntegerField(verbose_name="样本数")
    data_url = models.CharField(verbose_name="数据集存放的URL", max_length=100)
    data_file_size = models.FloatField(verbose_name="文件大小")
    data_file_name = models.CharField(verbose_name="数据文件名", max_length=100, default="")
    if_one_hot_finish = models.BooleanField(verbose_name="one-hot编码文件是否生成完成", default=False)


"""
数据集分箱表(data_set_bin)
ID
所属数据集ID
特征名
上界
下界
"""
class data_set_bin(models.Model):
    data_set_id = models.ForeignKey('data_set', on_delete=models.CASCADE, verbose_name="所属数据集ID")
    feature_name = models.CharField(verbose_name='特征名', max_length=100)
    bin_result = models.CharField(verbose_name='分箱情况', max_length=1000, default="")


"""
模型表(model_class)
ID
模型名
"""
class model_class(models.Model):
    model_name = models.CharField(verbose_name='模型名', max_length=100)


"""
模型运行记录表(model_run)
ID
模型ID
运行描述
开始时间
结束时间
训练状态
所用数据集ID
训练集与测试集的比例
划分的随机种子
"""
class task(models.Model):
    model_id = models.ForeignKey('model_class', on_delete=models.CASCADE, verbose_name="所用模型ID")
    task_name = models.CharField(verbose_name='任务名', max_length=100, default="")
    task_desc = models.CharField(verbose_name='模型运行描述', max_length=100, default="")
    task_start = models.DateTimeField(verbose_name='开始时间')
    task_end = models.DateTimeField(verbose_name='结束时间', default=None)
    task_status = models.IntegerField(verbose_name='训练状态', default=0) # 0:训练中 1:训练结束 2:训练失败
    data_set_id = models.ForeignKey('data_set', on_delete=models.CASCADE, verbose_name="所属数据集ID")
    model_run_train_test_split = models.FloatField(verbose_name='测试集与训练集比例')
    model_run_random_state = models.IntegerField(verbose_name='随机种子', default=None)
    if_use_one_hot = models.BooleanField(verbose_name='是否使用的one-hot数据集', default=False)


"""
线性模型训练结果记录表
ID
task_id
task_result
"""
class linear_task_result(models.Model):
    task_id = models.ForeignKey('task', on_delete=models.CASCADE, verbose_name="所属任务ID")
    task_result_str = models.CharField(verbose_name='任务结果字符串', max_length=10000, default="")
    task_result_url = models.CharField(verbose_name='任务结果url', max_length=1000, default="")
    accuracy = models.FloatField(verbose_name="准确率", default=-1)
    precision = models.FloatField(verbose_name="正确率", default=-1)
    recall = models.FloatField(verbose_name="召回率", default=-1)
    f1_score = models.FloatField(verbose_name="F1分数", default=-1)
    auc_plot_url = models.CharField(verbose_name="AUC图像存储URL", default="", max_length=1000)
