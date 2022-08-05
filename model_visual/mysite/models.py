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
数据Label列
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
    data_label = models.CharField(verbose_name="数据预测标签列的列名", max_length=100, default="")
    if_one_hot_finish = models.BooleanField(verbose_name="one-hot编码文件是否生成完成", default=False)
    special_code = models.CharField(verbose_name="缺失值标记", max_length=100, default="")

"""
数据集分箱表(data_set_bin)
ID
所属数据集ID
特征名
特征类型(离散/连续)
分箱结果的字符串
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
    model_if_linear = models.BooleanField(verbose_name='是否线性模型', default=False)
    task_name = models.CharField(verbose_name='任务名', max_length=100, default="")
    task_desc = models.CharField(verbose_name='模型运行描述', max_length=100, default="")
    task_start = models.DateTimeField(verbose_name='开始时间')
    task_end = models.DateTimeField(verbose_name='结束时间', default=None)
    task_status = models.IntegerField(verbose_name='训练状态', default=0) # 0:训练中 1:训练成功 2:训练失败
    data_set_id = models.ForeignKey('data_set', on_delete=models.CASCADE, verbose_name="所属数据集ID")
    model_run_train_test_split = models.FloatField(verbose_name='测试集与训练集比例')
    model_run_random_state = models.IntegerField(verbose_name='随机种子', default=None)
    task_error_log = models.CharField(verbose_name='训练错误信息', max_length=1000, default="")
    generated_dataset_id = models.IntegerField(verbose_name='如果使用的衍生数据集则保存衍生数据集ID，否则-1', default=-1)
    related_binning_task = models.IntegerField(verbose_name='相关的分箱任务，只有score_card类任务这个字段为非None', default=None)# models.ForeignKey('binning_task', on_delete=models.CASCADE, verbose_name="相关的分箱任务，只有score_card类任务这个字段为非None", default=None)

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

"""
分箱任务列表
数据集ID
任务状态
开始时间
结束时间
以该数据集的哪一列作为label来分箱的(因为是有监督分箱算法)
训练集比例
划分的随机种子
"""
class binning_task(models.Model):
    binning_task_name = models.CharField(verbose_name='分箱任务名', max_length=1000, default="")
    data_set_id = models.ForeignKey('data_set', on_delete=models.CASCADE, verbose_name="所属数据集ID")
    task_status = models.IntegerField(verbose_name='任务训练状态', default=None)
    task_start = models.DateTimeField(verbose_name='开始时间', default=None)
    task_end = models.DateTimeField(verbose_name='结束时间', default=None, null=True,blank=True)
    last_modified = models.DateTimeField(verbose_name='上次编辑时间', default=None, null=True,blank=True)
    training_label = models.CharField(verbose_name='训练的Label', max_length=1000, default="")
    train_ratio = models.FloatField(verbose_name='测试集与训练集比例', default=70)
    random_state = models.IntegerField(verbose_name='划分的随机种子', default=None)
    task_error_log = models.CharField(verbose_name='训练错误信息', max_length=1000, default="")
    pkl_file_url = models.CharField(verbose_name='pkl文件保存url', max_length=100, default="")

"""
分箱特征类
所属分箱任务ID
所属数据集ID
特征名
分箱数目
分箱结果字符串
"""
class binning_feature(models.Model):
    binning_task_id = models.ForeignKey('binning_task', on_delete=models.CASCADE, verbose_name="所属分箱任务ID")
    data_set_id = models.ForeignKey('data_set', on_delete=models.CASCADE, verbose_name="所属数据集ID")
    feature_name = models.CharField(verbose_name='特征名', max_length=100)
    bin_num = models.IntegerField(verbose_name='分箱数目', default=None)
    which_bin = models.IntegerField(verbose_name='第几个分箱', default=None)
    this_bin_str = models.CharField(verbose_name='当前分箱的分箱名', max_length=100, default="")
    count = models.IntegerField(verbose_name='该分箱内的样本数', default=0)
    count_percent = models.FloatField(verbose_name='占总样本', default=0)
    non_event_num = models.IntegerField(verbose_name='分箱内好样本数', default=0)
    event_num = models.IntegerField(verbose_name='分箱内坏样本数', default=0)
    event_rate = models.FloatField(verbose_name='坏样本比例', default=0)
    woe = models.FloatField(verbose_name='WoE值', default=0)
    this_iv = models.FloatField(verbose_name='IV值', default=0)
    total_iv = models.FloatField(verbose_name='该特征的IV值', default=0)
    is_special_values = models.BooleanField(verbose_name='是否是特殊值分箱', default=False)
    breaks = models.CharField(verbose_name='当前分箱的分箱名', max_length=100, default="")


"""
衍生数据集类
衍生数据集名称
参考的分箱结果（binning_task）
原数据集（data_set）
衍生类型：one-side interval衍生（需要指定单调性要求）、one-hot、WoE值替换。
"""
class generate_data_set_task(models.Model):
    generate_data_set_name = models.CharField(verbose_name='衍生数据集名称', max_length=100)
    binning_task = models.ForeignKey('binning_task', on_delete=models.CASCADE, verbose_name="参考的分箱结果")
    data_set = models.ForeignKey('data_set', on_delete=models.CASCADE, verbose_name="原数据集")
    generate_type = models.CharField(verbose_name='衍生类型', max_length=100)
    url = models.CharField(verbose_name='衍生数据集存储的地址url', max_length=100, default="")
    generate_status = models.IntegerField(verbose_name='生成状态', default=0)
    mono = models.CharField(verbose_name='单调性', max_length=1000, default="")