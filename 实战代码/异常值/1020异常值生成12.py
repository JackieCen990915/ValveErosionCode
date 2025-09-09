
import numpy as np
import random

'''
为了确保代码模型的复现性，就统一用这个随机数
'''

#random.seed(42)  # 设置随机种子，这里的 42 可以是任何你想要的整数，只要种子相同，生成的随机序列就相同。
random.seed(433)

##随机索引
##### 将异常值添加到训练集-----60*10%=6

# 所有下标。使用range()函数生成从0到74的列表，间隔为1  
all_index = list(range(0, 75))  # 注意：range的结束值是排他性的，所以要写75而不是74

#测试集下标
test_index = [3, 5, 8, 17, 20, 21, 27, 35, 41, 42, 45, 46, 50, 52, 59]
  
# 所有下标-测试集下标=训练集下标。使用filter函数和lambda表达式去除特定的数字  
train_index = list(filter(lambda num: num not in test_index, all_index))  

# 随机选择12个训练集下标，之后会修改这6个下标的数值，以此生成噪声数据
random_selected_train_index = random.sample(train_index, 12)

print("all_index:",all_index)
print("test_index:",test_index)
print("train_index:",train_index)
print("random_selected_train_index:",random_selected_train_index)

'''
all_index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74]
test_index: [3, 5, 8, 17, 20, 21, 27, 35, 41, 42, 45, 46, 50, 52, 59]
train_index: [0, 1, 2, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 43, 44, 47, 48, 49, 51, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74]
'''


##随机倍数----相对误差大于30%。
'''
random_multiple1=np.random.uniform(0, 0.7, (6, 1))
random_multiple2=np.random.uniform(1.3, 5, (6, 1))
print('random_multiple11:',random_multiple1)
print('random_multiple12:',random_multiple2)
'''


random_multiple=np.random.uniform(1.3, 5, (12, 1))
print('random_multiple:',random_multiple)




'''

'''

