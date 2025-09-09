
import numpy as np
import random

'''
为了确保代码模型的复现性，就统一用这个随机数
'''

#random.seed(42)  # 设置随机种子，这里的 42 可以是任何你想要的整数，只要种子相同，生成的随机序列就相同。
random.seed(42)

##随机索引
##### 将异常值添加到训练集-----60*10%=6

# 所有下标。使用range()函数生成从0到70的列表，间隔为1  
all_index = list(range(0, 70))  # 注意：range的结束值是排他性的，所以要写70而不是69


'''
---------------y_test------------------
22    0.015090
0     0.008190
49    0.003840
4     0.020600
54    0.000491
18    0.007440
10    0.023600
33    0.006770
45    0.000186
12    0.007080
31    0.019250
9     0.014050
67    0.000063
5     0.007710
'''


#测试集下标
test_index = [22, 0, 49, 4, 54, 18, 10, 33, 45, 12, 31, 9, 67, 5]
  
# 所有下标-测试集下标=训练集下标。使用filter函数和lambda表达式去除特定的数字  
train_index = list(filter(lambda num: num not in test_index, all_index))  

# 随机选择6个训练集下标，之后会修改这6个下标的数值，以此生成噪声数据
random_selected_train_index = random.sample(train_index, 6)

print("all_index:",all_index)
print("test_index:",test_index)
print("train_index:",train_index)
print("random_selected_train_index:",random_selected_train_index)



##随机倍数----相对误差大于30%。
'''
random_multiple1=np.random.uniform(0, 0.7, (3, 1))
random_multiple2=np.random.uniform(1.3, 2, (3, 1))
print('random_multiple11:',random_multiple1)
print('random_multiple12:',random_multiple2)
'''
random_multiple=np.random.uniform(1.3, 3, (6, 1))
print('random_multiple:',random_multiple)




'''
all_index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]
test_index: [22, 0, 49, 4, 54, 18, 10, 33, 45, 12, 31, 9, 67, 5]
train_index: [1, 2, 3, 6, 7, 8, 11, 13, 14, 15, 16, 17, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69]
random_selected_train_index: [52, 13, 2, 60, 25, 23]
random_multiple: [[2.08767221]
 [1.3688075 ]
 [2.00090929]
 [2.65562694]
 [1.42969978]
 [2.32287012]]
'''


