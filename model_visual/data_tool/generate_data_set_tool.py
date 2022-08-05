
import pandas as pd

# 用于衍生变量的函数
# 衍生规则:
# 以ExternalRiskEstimate为例，因为是与label之间关系为单调递减的，因此衍生变量为：是否<35, 是否<59.5, ... ,是否<87.5，是否<Inf，是否有-7类型的missing，是否有-8类型的missing，是否有-9类型的missing【都是二值0或者1】
# 如果要求是单调增，则将<号全部改为>=即可
# 参数解释：
#   feature：该特征值，split: 分割列表,  mono: 0不要求单调 1单调递减 2单调递增 为None则默认为0
# 返回值：
#  返回该特征拆分为几个虚拟变量后得到的一个二维pd.dataframe
def trans_into_dummy_variables(feature, split, mono, numerical, special_code):
    if mono == None:
        mono = 0
    if numerical == True:  # 连续变量
        # 将split改为int
        split = [float(s) for s in split]
        res_arr = []
        col_names = generate_col_name_list(feature.name, split, mono, numerical, special_code)
        col_len = len(col_names)

        for f in feature: # 遍历这个特征的每个值，把这一列衍生为多列的过程
            if special_code != None:
                row = [0 for i in range(len(split) + 1 + len(special_code))]
            else:
                row = [0 for i in range(len(split) + 1)]
            # 如果这个数据集是有special_code的 并且 当前遍历的特征值是属于special_code 则直接把这个special_code箱置1即可
            if special_code != None and str(f) in special_code:
                for i in range(len(special_code)): # 0 1 2
                    if str(f) == special_code[i]:
                        row[len(split) + i + 1] = 1
                        break # 应该不会出现同时满足两个分箱的情况，如果后续有这种情况，把这个break取消即可
            else:
                # 填充各列的值
                if mono == 1:  # 单减 <
                    for i in range(len(split) - 1, -1, -1):
                        if f < split[i]:
                            row[i] = 1
                        else:
                            break
                elif mono == 2:  # 单增 >=
                    for i in range(len(split)):
                        if f >= split[i]:
                            row[ i +1] = 1
                        else:
                            break
                elif mono == 0:  # 无单调约束的
                    for i in range(len(split)):
                        if i == 0 and f < split[i]: # 小于第一个
                            row[i] = 1
                            break
                        elif i == len(split ) -1 and f >= split[i]: # 大于最后一个  3 4 5 7    <3 3-4  4-5  5-7 >7
                            row[len(split)] = 1
                            break
                        else:
                            if f >= split[i - 1] and f < split[i]:
                                row[i] = 1
                                break

                # 填充有单调约束的衍生边缘列 <+INF  >-INF
                if mono == 1: # 3 4 5 7   <3 <4 <5 <7 <+inf -7 -8 -9
                    row[len(split)] = 1
                elif mono == 2: # 3 4 5 7 >-inf >3 >4 >5 >7 -7 -8 -9
                    row[0] = 1

            res_arr.append(row)
        res_df = pd.DataFrame(res_arr, columns=col_names)
        # 保留原数据集的序号index
        res_df.index = feature.index
        return res_df


def generate_col_name_list(feature_name, split, mono, numerical, special_code):
    if mono is None:
        mono = 0
    if numerical is True:  # 连续变量
        res = []

        count = 0
        for s in split:
            if mono == 1:  # 单减约束
                res.append(feature_name + "_" + str(count + 1) + " <" + str(s))
            elif mono == 2:  # 单增约束
                res.append(feature_name + "_" + str(count + 1) + " >=" + str(s))
            elif mono == 0:  # 没有单调性约束的变量，直接one-hot衍生即可
                if count == 0:
                    res.append(feature_name + "_" + str(count + 1) + " <" + str(s))
                else:
                    res.append(feature_name + "_" + str(count + 1) + " " + str(split[count - 1]) + "<=X<" + str(s))
            count = count + 1

        if mono == 1:
            res.append(feature_name + " < +INF")
        elif mono == 2:
            res.insert(0, feature_name + " > -INF")
        elif mono == 0:
            res.append(feature_name + " >=" + str(split[-1]))
        if special_code is not None:
            for i in range(len(special_code)):
                res.append(feature_name + "_" + special_code[i])

    return res


def one_side_interval(X, features, features_splits, features_mono, special_code):
    # 生成所有特征的衍生变量，汇总到X_train_final中
    generated_data = pd.DataFrame()
    for f in features:
        temp_res = trans_into_dummy_variables(X[f], features_splits[f], mono=features_mono.get(f ,None), numerical=True, special_code=special_code)

        if generated_data.empty:
            generated_data = temp_res
        else:
            generated_data = pd.concat([generated_data, temp_res], axis=1, ignore_index=False)
    return generated_data
