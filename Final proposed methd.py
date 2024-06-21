from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix 
import cv2
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

im = cv2.imread("C:/Users/ayush/OneDrive/Desktop/conference paper/1.jpg", 0)
im2 = cv2.imread("C:/Users/ayush/OneDrive/Desktop/conference paper/2.jpg", 0)
im = cv2.copyMakeBorder(im, 2, 2, 2, 2, cv2.BORDER_CONSTANT)
im2 = cv2.copyMakeBorder(im2, 2, 2, 2, 2, cv2.BORDER_CONSTANT)
image=cv2.imread("C:/Users/ayush/OneDrive/Desktop/conference paper/gt.jpg", 0)
value=[255,255,255]
image2 = cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, None,value)

def diff1(i,j,k,l):
    global sum1
    sum1=abs(int(im[i][j])-int(im[k][l]))
    
def diff2(i,j,k,l):
    global sum1
    sum1=abs(int(im2[i][j])-int(im2[k][l]))

def fun(i,j):
    a1=[]
    b1=[]
    result=[]
    m=0
    for k in range(-2,3):
        for l in range(-2,3):
            global sum1
            sum1=0
            diff1(i,j,i+k,j+l)
            Th.append(sum1)
            if(sum1!=0):
                m=m+1
            sum1=0
            diff2(i,j,i+k,j+l)
            Th2.append(sum1)
            if(sum1!=0):
                m=m+1
    array=Th+Th2
    
    return array
  

r=image2.shape[0]
c=image2.shape[1]


changed_indices=[]
unchanged_indices=[]
for i in range(2,r-2):
    for j in range(2,c-2):
        if(image2[i][j]==0):
            changed_indices.append([i,j])
        else:
            unchanged_indices.append([i,j])

X1=[]
X2=[]
X_train =[]
X_test=[] 
y_train=[] 
y_test=[]
global sum1
sum1=0

for j in range(len(changed_indices)):
    Th = []
    Th2 = []
    X1.append(fun(changed_indices[j][0], changed_indices[j][1]))
for j in range(len(unchanged_indices)):
    Th = []
    Th2 = []
    X2.append(fun(unchanged_indices[j][0], unchanged_indices[j][1]))

rand_x1=random.sample(X1, k=150)
rand_x2=random.sample(X2, k=150)
X_train = rand_x1+rand_x2
y_train = np.concatenate((np.zeros(150),np.ones(150)))

rand_x3=list(filter(lambda x: x not in rand_x1, X1))
rand_x4=list(filter(lambda x: x not in rand_x2, X2))
X_test = X1+X2
y_test= np.concatenate((np.zeros(len(X1)),np.ones(len(X2))))
ans_x=[]
ans_y=[]
for i in range(2,r-2):
    for j in range(2,c-2):
        Th = []
        Th2 = []
        ans_x.append(fun(i,j))
        if(image2[i][j]==0):
            ans_y.append(0)
        else:
            ans_y.append(1)
ans_x=np.array(ans_x,dtype=object)
ans_y=np.array(ans_y)
svm = SVC(kernel='linear')

#PSO optimization->

X=[]
for i in range(10):
    rand_x1=random.sample(X1, k=150)
    rand_x2=random.sample(X2, k=150)
    temp = rand_x1+rand_x2
    X.append(temp)

"""
with open('R1c_X.pkl', 'wb') as f:
    # write the X to the file as pickle
    pickle.dump(X, f)

X = []
with (open('R1c_X.pkl', "rb")) as openfile:
    while True:
        try:
            X=pickle.load(openfile)
        except EOFError:
            break
"""
Y=np.concatenate((np.zeros(150),np.ones(150)))
Y_val=np.concatenate((np.zeros(25),np.ones(25)))

rand_x1=random.sample(X1, k=25)
rand_x2=random.sample(X2, k=25)
validation = rand_x1+rand_x2

"""
with open('R1c_val.pkl', 'wb') as f:
    # write the validation set to the file as pickle
    pickle.dump(validation, f)


with (open('R1c_val.pkl', "rb")) as openfile:
    while True:
        try:
            validation=pickle.load(openfile)
        except EOFError:
            break
"""

v=[0,0,0,0,0,0,0,0,0,0]
P_best=X
fit_best=[]
for i in range(10):
    svm.fit(X[i], Y)
    y_pred=svm.predict(validation)
    accuracy=accuracy_score(Y_val,y_pred)
    fit_best.append(accuracy)
gbest=X[0]
gbest_fit=fit_best[0]
g_fitness=[0,0,0,0,0,0,0,0,0,0]
for t in range(20):

    print(t+1)
    for i in range(10):
        svm.fit(X[i], Y)
        print(i,end=" ")
        y_pred=svm.predict(validation)
        accuracy=accuracy_score(Y_val,y_pred)
        g_fitness[i]=accuracy
        if(gbest_fit<g_fitness[i]):
            gbest=X[i]
            gbest_fit=g_fitness[i]
        if(fit_best[i]<g_fitness[i]):
            fit_best[i]=g_fitness[i]
            P_best[i]=X[i]
    for i in range(10):
        r1=random.random()
        r2=random.random()
        v[i]=0.5*v[i]+2*r1*(np.array(P_best[i])-np.array(X[i]))+2*r2*(np.array(gbest)-np.array(X[i]))
        X[i]=(np.array(v[i])+np.array(X[i]))



Y=np.concatenate((np.zeros(150),np.ones(150)))


print("AFTER OPTIMIZATION->")

objects=gbest

svm.fit(objects, Y)
y_pred = svm.predict(X_train)
# Calculate train accuracy
accuracy = accuracy_score(y_train, y_pred)
print("Train Accuracy:", accuracy)
y_pred = svm.predict(ans_x)
# Calculate accuracy
accuracy = accuracy_score(ans_y, y_pred)
print("Test Accuracy:", accuracy)

print(np.count_nonzero(y_pred == 1))
out = np.array(y_pred, dtype=np.uint8).reshape((200,200))
new=((out/1)*255.9).astype(np.uint8)
#new.save('D:\\VIT\\Conference paper\\Proposed Paper\\r1.jpg')
#cv2.imwrite("D:\VIT\Conference paper\Reasarch Images\Data Set\g1.tif", new[ 3:r-3, 3:c-3 ]   )
cv2.imwrite("D:\VIT\Conference paper\Reasarch Images\Data Set\\r2_5.tif", new )
plt.subplot(221), plt.imshow(im,'gray'),plt.title("image1 "),plt.axis('off')
plt.subplot(222), plt.imshow(im2,'gray'),plt.title("image2"),plt.axis('off')
plt.subplot(223), plt.imshow(new,'gray'),plt.title("image ch"),plt.axis('off')
plt.subplot(224), plt.imshow(image,'gray'),plt.title("image GT"),plt.axis('off')
plt.show()