import os
import glob
from sklearn.model_selection import train_test_split
import shutil

q1_dir='D:/dataset'
print(os.path.exists(q1_dir))
train_num=2000
eval_num=500
test_num=500
# D:\research\testing samples.zip\testing samples  copy
cube_folders=glob.glob(os.path.join(q1_dir, '*/'))

X_train, remaining=train_test_split(cube_folders, train_size=train_num,
                                 random_state=42)

X_eval, remaining=train_test_split(remaining, train_size=eval_num,
                                 random_state=43)

X_test, remaining=train_test_split(remaining, train_size=test_num,
                                 random_state=44)

train_dir=os.path.join(q1_dir,'train')
os.makedirs(train_dir, exist_ok=True)
eval_dir=os.path.join(q1_dir,'eval')
os.makedirs(eval_dir, exist_ok=True)
test_dir=os.path.join(q1_dir,'test')
os.makedirs(test_dir, exist_ok=True)


for fp in X_train:
    shutil.move(fp, os.path.join(train_dir))

for fp in X_eval:
    shutil.move(fp, os.path.join(eval_dir))

for fp in X_test:
    shutil.move(fp, os.path.join(test_dir))