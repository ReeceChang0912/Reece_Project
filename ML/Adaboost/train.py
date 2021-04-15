from PIL import Image
import os
import numpy as np
from feature import NPDFeature

try:
    import cPickle as pickle
except ImportError:
    import pickle
import ensemble 
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

def load_image(face_filepath,nonface_filepath):  
    #os.listdir(filename)返回filename中所有文件的文件名列表
    face_imgs = os.listdir(face_filepath)
    nonface_imgs = os.listdir(nonface_filepath)
    num = len(face_imgs)
    datasets=[]
    print("loading image...")
    for i in range(num):
        img=Image.open(face_filepath+face_imgs[i])
        img=img.convert("L")
        img=img.resize((24,24), resample=Image.LANCZOS)
        arr = np.asarray(img)
        datasets.append([NPDFeature(arr).extract(),1])
              
        img=Image.open(nonface_filepath+nonface_imgs[i])
        img=img.convert("L")
        img=img.resize((24,24), resample=Image.LANCZOS)
        arr = np.asarray(img)
        datasets.append([NPDFeature(arr).extract(),-1])
        
    print("Successfully loaded image!!!") 
    f = open('dump.txt', 'wb')
    pickle.dump(datasets,f)
    f.close()

def get_datasets(face_filepath,nonface_filepath):
    datasets=[]
    if os.path.isfile('dump.txt'):
        f = open('dump.txt', 'rb')
        datasets = pickle.load(f)
        f.close()
    else:
        load_image(face_filepath,nonface_filepath)
        f = open('dump.txt', 'rb')
        datasets  = pickle.load(f)
        f.close()
    return datasets


from sklearn.model_selection import train_test_split
def main():
   face_filepath = "./datasets/original/face/"
   nonface_filepath = "./datasets/original/nonface/"
   datasets=get_datasets(face_filepath,nonface_filepath)

   X=[]
   y=[]
   for i in range(len(datasets)):
    X.append(datasets[i][0])
    y.append(datasets[i][1])

   X_train,X_test,y_train,y_test=train_test_split(np.array(X),np.array(y),test_size=0.3,shuffle=True)
   y_train=y_train.reshape(y_train.shape[0],1)
   y_test=y_test.reshape(y_test.shape[0],1)
   print(X_train.shape)
   print(y_train.shape)
   print(X_test.shape)
   print(y_test.shape)

   if os.path.isfile('report16.txt'):
    f = open('report16.txt', 'r')
    string=f.read()
    print("result:")
    print(string)
    f.close()
   else:
    model=ensemble.AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),16)
    model.fit(X_train,y_train)
    y_predict=model.predict(X_test,0)
    #print("y_predict:",y_predict)
    string=classification_report(y_test,y_predict,target_names=["nonface","face"],digits=4)
    print("result:")
    print(string)
    file=open("report.txt","w")
    file.write(string)
    file.close()

if __name__=="__main__":
        main()
