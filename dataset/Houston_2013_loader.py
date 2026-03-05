from scipy.io import loadmat

mat_hl = loadmat('data/raw/Houston 2013/Houston_2013_data_hl.mat')

print(mat_hl.keys())
for k,v in mat_hl.items():
    if not k.startswith('__'):
        print(k, type(v), getattr(v, "shape", None))   # 变量名、类型、shape
        # 'data' (HSI + LiDAR), 'AllTrueLabel (gt)', 'train_select_index', 'trainall'
data_hl = mat_hl["data"]
print(data_hl.shape)   # (349, 1905, 145) 144 HSI + 1 LiDAR
print(mat_hl["train_select_index"].shape)  # (13, 20)
print(mat_hl["trainall"].shape)   #(2, 15029)

# mat_gt = loadmat('data/raw/Houston 2013/Houston_2013_gt.mat')
# print(mat_gt.keys())
# for k,v in mat_gt.items():
#     if not k.startswith('__'):
#         print(k, type(v), getattr(v, "shape", None))   

# data_gt = mat_gt["label_data"]  # (349, 1905)


# mat = loadmat('data/raw/Houston 2013/Houston_2013.mat')
# print(mat.keys())
# for k,v in mat.items():
#     if not k.startswith('__'):
#         print(k, type(v), getattr(v, "shape", None))    
        
# data = mat["cube_data"]   #  (349, 1905, 144) 144 HSI