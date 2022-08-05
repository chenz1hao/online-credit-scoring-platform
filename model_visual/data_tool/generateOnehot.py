import pandas as pd

# 生成one-hot某一行的字典格式数据，用于追加到dataframe中，在生成one-hot表时用
def generateCurRowInDict(col_names, whichDivision):
    res = {}
    hasSetOne = False

    for col_name in col_names:
        if hasSetOne:
            res[col_name] = 0
        else:
            if str(whichDivision) in col_name:
                res[col_name] = 1
                hasSetOne = True
            else:
                res[col_name] = 0

    return res

# 判断当前值处于第几个区间中,在生成one-hot表时用
def checkWhichDivision(split_list, var):
    for i in range(len(split_list)):
        begin_index = split_list[i].index('(')
        mid_index = split_list[i].index(',')
        if('-INF' in split_list[i]): # 含有-INF说明是第一个区间，则直接取'逗号'开始至']'结束
            end_index = split_list[i].index(']')
            # print(split_list[i][mid_index+1:end_index])
            if (var <= float(split_list[i][mid_index+1 : end_index])):
                return i+1
        elif('+INF' in split_list[i]): # 含有+INF说明是最后一个区间，则直接取第一个括号开始至','结束
            # print(split_list[i][begin_index+1:mid_index])
            if (var > float(split_list[i][begin_index+1:mid_index])):
                return i+1
        else:   # 处于中间区间的，取两个数，第一个数是'('开始','结束，第二个数是','开始']'结束
            end_index = split_list[i].index(']')
            if float(split_list[i][mid_index+1:end_index]) >= var > float(split_list[i][begin_index+1: mid_index]):
                return i+1


# 生成表头 例如var=xx split_num=3 则返回 ['xx_1', 'xx_2', 'xx_3']，在生成one-hot表时用
def generateColNames(var, split_num):
    res = []
    for i in range(split_num):
        res.append(var + '_' + str(i+1))
    return res

# 根据传入的subscale的变量list来生成one-hot文件
def generateOneHotByList(data_url, list, var_split_list, generatePath):
    # 取出原数据集中对应列
    data = pd.read_csv(data_url)
    try:
        partial_data = data[list] # 只包含在了该subscale中的变量的部分data视图
    except KeyError:
        raise Exception('配置文件中的变量不能在原数据集中找到，检查subscale.xml中的变量')


    dataframe_final = pd.DataFrame()

    for var in list:
        if var in var_split_list.keys(): # 确定是否要分箱
            cur_split_list = var_split_list[var]
            split_num = len(cur_split_list)
            col_names = generateColNames(var, split_num)
            dataframe_temp = pd.DataFrame(columns = col_names) # 有几个分段就要创建一个几列的dataframe
            cur_col = partial_data[var]
            # 这个for循环一行一行插入dataframe_temp
            for index in cur_col.index:
                whichDivision = checkWhichDivision(cur_split_list, cur_col[index]) # 判断当前这个值在哪个区间里面
                temp_row = generateCurRowInDict(col_names, whichDivision) # 根据所处区间生成一个字典数据，如："{'ExternalRiskEstimate_1': 0, 'ExternalRiskEstimate_2': 0, 'ExternalRiskEstimate_3': 1}"用于之后的追加
                dataframe_temp = dataframe_temp.append(temp_row, ignore_index = True)
            dataframe_temp.columns = col_names

            # 合并在dataframe_final中
            if dataframe_final.empty:
                dataframe_final = dataframe_temp
            else:
                dataframe_final = pd.concat([dataframe_final, dataframe_temp], axis = 1, ignore_index = False)

    dataframe_final.insert(0, data.columns.values[0], data[data.columns.values[0]])
    dataframe_final.to_csv(generatePath, index=0)
    print('生成', generatePath, '成功')