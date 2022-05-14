import os
import numpy as np
import matplotlib.pyplot as plt
import urllib.request


f = open("classes.txt","r")
classes = f.readlines()
f.close()

classes = [c.replace('\n','').replace(' ','_') for c in classes]
print(classes)

def download():
  base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
  for c in classes:
    cls_url = c.replace('_', '%20')
    path = base+cls_url+'.npy'
    print(path)
    urllib.request.urlretrieve(path, 'data/'+c+'.npy')
    
#download()

if not os.path.exists('data'):
    os.makedirs('data')

dataset_path = 'data/'
files = os.listdir(dataset_path)

os.makedirs(dataset_path + 'test/')
os.makedirs(dataset_path + 'valid/')
os.makedirs(dataset_path + 'train/')

for f in files:
  print(f)
  dataset = np.load(dataset_path+f)
  np.random.shuffle(dataset)
  dataset = dataset[:35000]
  test, val, train = np.split(dataset, [int(0.2*len(dataset)), int(0.44*len(dataset))])
  np.save(dataset_path + 'test/' + f, test)
  np.save(dataset_path + 'valid/' + f, val)
  np.save(dataset_path + 'train/' + f, train)


check = np.load(dataset_path + 'angel.npy')


print(check.shape)

plt.figure(figsize=(2,2))
plt.imshow(check[4].reshape(28,28))



