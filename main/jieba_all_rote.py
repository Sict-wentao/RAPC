import jieba
import copy

# 测试jieba全模式输出结果
test_str = '我毕业于北京大学'
#split_str = jieba.cut(test_str, cut_all=True)
test_dag  = jieba.get_DAG(test_str)
print(len(test_dag))
print(test_dag)

all_list = []
# 输出当前有向无环图的所有的路径，通过DFS遍历所有的路径
def dag_all(cur, cur_list, dag, seq_len):
    if cur not in dag:
        return
    temp_list = dag[cur]
    # 遍历temp_list中的任何一个点作为起始结点
    for i, index in enumerate(temp_list):
        if i == 0:
            now_list = [index]
        else:
            now_list = [temp_list[0], index]
        new_cur_list = copy.deepcopy(cur_list)
        new_cur_list.append(now_list)
        if index == seq_len-1:
            all_list.append(new_cur_list)
            return
        dag_all(index+1, new_cur_list, dag, seq_len)

dag_all(0, [], test_dag, len(test_dag))

for str_list in all_list:
    print(str_list)
    split_str_list = []
    for token in str_list:
        if len(token) == 1:
            split_str_list.append(test_str[token[0]])
        else:
            split_str_list.append(test_str[token[0]:token[1]+1])
    print('切分结果: ', split_str_list)