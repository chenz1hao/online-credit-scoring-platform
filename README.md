# 基于Django的信用评分模型管理系统
支持在线数据集管理、特征分箱、在线统计机器学习模型异步训练（目前只有逻辑回归，后序还可以添加各类不同统计学习模型）、在线模型性能评估（包括Accuracy、Recall、Precision、F1-Score）等功能，后续还会加入可解释机器学习技术的嵌入，目的在于提高模型决策的可解释与可信

依赖包见requirement.txt
venv为本地虚拟环境，一般建议克隆下来后自己新建个环境
Python版本 3.8.6
异步训练采用Celery+Redis 需要自行修改redis的host:port等
