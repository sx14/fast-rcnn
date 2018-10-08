# list_path = '/media/sunx/Data/dataset/visual genome/feature/object/label/train.txt'
#
# with open(list_path, 'r') as t_list:
#     lines = t_list.read().splitlines()
#
# p = 0
# n = 0
#
# for l in lines:
#     info = l.split(' ')
#     flag = int(info[3])
#     if flag > 0:
#         p += 1
#     else:
#         n += 1
#
# print('positive: %d | negative: %d' % (p,n))