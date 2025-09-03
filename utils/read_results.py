from glob import glob

path = '/mnt/orton/codes/6_code_noise/my_baseline/models_final/'
txts = glob(path + '/*72*/*.txt')
txts.sort()
for txt in txts:
    with open(txt,'r') as  file:
        aa = file.readlines()
        result_use = aa[-1]
        print(txt,result_use)
        # print('fds')

# print('fds')