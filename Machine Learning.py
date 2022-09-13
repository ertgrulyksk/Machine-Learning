#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[3]:


data = pd.read_csv("ortopedik_hastaların_biyomekanik_özellikleri.csv")
data.head()


# In[4]:


sns.countplot(data["class"])
plt.show()


# In[5]:


data["class"] = [1 if each == "Abnormal" else 0 for each in data["class"]]
data.head()


# In[6]:


data.info()


# In[7]:


y = data["class"].values
x_data = data.drop(["class"],axis=1)


# In[8]:


sns.pairplot(x_data)
plt.show()


# In[9]:


x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.15,random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T


print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)


# In[12]:


from sklearn.linear_model import LogisticRegression


# In[13]:


lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)


# In[14]:


test_dogrulugu = lr.score(x_test.T,y_test.T)
print("Test Doğruluğu: {}".format(test_dogrulugu))


# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


data = pd.read_csv("ortopedik_hastaların_biyomekanik_özellikleri.csv")
data.head(3)


# In[17]:


sns.scatterplot(data=data, x="lumbar_lordosis_angle", y="pelvic_tilt numeric", hue="class")
plt.xlabel("lomber lordoz açısı")
plt.ylabel("pelvik eğim")
plt.legend()
plt.show()


# In[18]:


data["class"] = [1 if each == "Abnormal" else 0 for each in data["class"]]
data.head(3)


# In[19]:


y = data["class"].values
x_data = data.drop(["class"],axis=1)


# In[20]:


x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.15,random_state=1)


# In[23]:


from sklearn.neighbors import KNeighborsClassifier
komsu_sayisi = 4
knn = KNeighborsClassifier(n_neighbors = komsu_sayisi)
knn.fit(x_train,y_train)

prediction = knn.predict(x_test)
print(" {} En Yakın Komşu Modeli Test Doğruluk: {} ".format(komsu_sayisi,knn.score(x_test,y_test)))


# In[25]:


score_list = []
for each in range(1,50):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,50),score_list)
plt.xlabel("k değerleri")
plt.ylabel("Doğruluk")
plt.title("En iyi K Değerinin Bulunması")
plt.show()


# In[26]:


import pandas as pd
import numpy as np


# In[27]:


data = pd.read_csv("ortopedik_hastaların_biyomekanik_özellikleri.csv")
data.head(3)


# In[28]:


data["class"] = [1 if each == "Abnormal" else 0 for each in data["class"]]
data.head(3)


# In[29]:


y = data["class"].values
x_data = data.drop(["class"],axis=1)


# In[30]:


x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[31]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.15,random_state=1)


# In[32]:


from sklearn.svm import SVC

svm = SVC(random_state = 1)
svm.fit(x_train,y_train)

print("Destek Vektör Makinesi Modeli Test Doğruluk: {}".format(svm.score(x_test,y_test)))


# In[34]:


import pandas as pd
import numpy as np


# In[35]:


data = pd.read_csv("ortopedik_hastaların_biyomekanik_özellikleri.csv")
data["class"] = [1 if each == "Abnormal" else 0 for each in data["class"]]
data.head(3)


# In[36]:


y = data["class"].values
x_data = data.drop(["class"],axis=1)


# In[37]:


x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[38]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.15,random_state=1)


# In[44]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

print("Kara Ağacı Modeli Test Doğruluk: {}".format(dt.score(x_test,y_test)))


# In[46]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100,random_state = 1)
rf.fit(x_train,y_train)

print("Rastgele Orman Modeli Test Doğruluk: {}".format(dt.score(x_test,y_test)))


# In[47]:


#%% confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
cm


# In[50]:


f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="white",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_pred")
plt.show()


# In[52]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[53]:


x1 = np.random.normal(25,5,1000)
y1 = np.random.normal(25,5,1000)

x2 = np.random.normal(55,5,1000)
y2 = np.random.normal(60,5,1000)

x3 = np.random.normal(55,5,1000)
y3 = np.random.normal(15,5,1000)

x = np.concatenate((x1,x2,x3),axis = 0)
y = np.concatenate((y1,y2,y3),axis = 0)

dictionary = {"x":x,"y":y}

data = pd.DataFrame(dictionary)
data.head()


# In[54]:


plt.figure()
plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.xlabel("x")
plt.ylabel("y")
plt.title("K-Ortalama Kümeleme Yöntemi İçin Oluşturulan Veri Seti")
plt.show()


# In[55]:


# k ortlama algoritması veriyi böyle görecek
plt.figure()
plt.scatter(x1,y1,color = "black")
plt.scatter(x2,y2,color = "black")
plt.scatter(x3,y3,color = "black")
plt.xlabel("x")
plt.ylabel("y")
plt.title("K-Ortalama Kümeleme Yöntemi İçin Oluşturulan Veri Seti")
plt.show()


# In[56]:


from sklearn.cluster import KMeans
wcss = []

for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

plt.figure()    
plt.plot(range(1,15),wcss)
plt.xticks(range(1,15))
plt.xlabel("Küme Sayısı (K)")
plt.ylabel("wcss")
plt.show()


# In[58]:


k_ortalama = KMeans(n_clusters=3)
kumeler = k_ortalama.fit_predict(data)

data["label"] = kumeler

plt.figure()
plt.scatter(data.x[data.label == 0],data.y[data.label == 0],color = "red", label = "Kume 1")
plt.scatter(data.x[data.label == 1],data.y[data.label == 1],color = "green", label = "Kume 2")
plt.scatter(data.x[data.label == 2],data.y[data.label == 2],color = "blue", label = "Kume 3")
plt.scatter(k_ortalama.cluster_centers_[:,0],k_ortalama.cluster_centers_[:,1],color = "yellow")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("3-Ortalama Kümeleme Sonucu")
plt.show()


# In[59]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[60]:


x1 = np.random.normal(25,5,20)
y1 = np.random.normal(25,5,20)

x2 = np.random.normal(55,5,20)
y2 = np.random.normal(60,5,20)

x3 = np.random.normal(55,5,20)
y3 = np.random.normal(15,5,20)

x = np.concatenate((x1,x2,x3),axis = 0)
y = np.concatenate((y1,y2,y3),axis = 0)

dictionary = {"x":x,"y":y}

data = pd.DataFrame(dictionary)
data.head()


# In[61]:


plt.figure()
plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Hiyerarşik Kümeleme Yöntemi İçin Oluşturulan Veri Seti")
plt.show()


# In[64]:


# %% demdogram
from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(data,method="ward")
dendrogram(merg,leaf_rotation = 90)
plt.xlabel("Veri Noktaları")
plt.ylabel("Öklid Mesafesi")
plt.show()


# In[65]:


from sklearn.cluster import AgglomerativeClustering

hiyerarsi_kume = AgglomerativeClustering(n_clusters = 3,affinity = "euclidean", linkage = "ward")
kume = hiyerarsi_kume.fit_predict(data)

data["label"] = kume

plt.figure()
plt.scatter(data.x[data.label == 0],data.y[data.label == 0],color = "red", label = "Kume 1")
plt.scatter(data.x[data.label == 1],data.y[data.label == 1],color = "green", label = "Kume 2")
plt.scatter(data.x[data.label == 2],data.y[data.label == 2],color = "blue", label = "Kume 3")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("3-Ortalama Kümeleme Sonucu")
plt.show()


# In[67]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.datasets import load_iris


# In[68]:


iris = load_iris()

data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data,columns = feature_names)
df["sinif"] = y
df.head()


# In[70]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 2,whiten = True ) 
pca.fit(data)

x_pca = pca.transform(data)

print("variance ratio: ", pca.explained_variance_ratio_)

print("sum: ",sum(pca.explained_variance_ratio_))


# In[71]:


df["p1"] = x_pca[:,0]
df["p2"] = x_pca[:,1]

color = ["red","green","blue"]

for each in range(3):
    plt.scatter(df.p1[df.sinif == each],df.p2[df.sinif == each],color = color[each],label = iris.target_names[each])

plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()


# In[72]:


from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

X, y = make_circles(n_samples=1_100, factor=0.3, noise=0.05, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)


# In[73]:


import matplotlib.pyplot as plt

_, (train_ax, test_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))

train_ax.scatter(X_train[:, 0], X_train[:,1], c=y_train)
train_ax.set_ylabel("Öznitelik #1")
train_ax.set_xlabel("Öznitelik #0")
train_ax.set_title("Eğitim Verisi")

test_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
test_ax.set_xlabel("Öznitelik #0")
_ = test_ax.set_title("Test Verisi")


# In[75]:


from sklearn.decomposition import PCA, KernelPCA

pca = PCA(n_components=2)
kernel_pca = KernelPCA(
    n_components=None, kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.1
)

X_test_pca = pca.fit(X_train).transform(X_test)
X_test_kernel_pca = kernel_pca.fit(X_train).transform(X_test)


# In[77]:


fig, (orig_data_ax, pca_proj_ax, kernel_pca_proj_ax) = plt.subplots(
    ncols=3, figsize=(14, 4)
)

orig_data_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
orig_data_ax.set_ylabel("Özitelik #1")
orig_data_ax.set_xlabel("Özitelik #0")
orig_data_ax.set_title("Test Verisi")

pca_proj_ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test)
pca_proj_ax.set_ylabel("Temel Bileşen #1")
pca_proj_ax.set_xlabel("Temel Bileşen #0")
pca_proj_ax.set_title("Test Verisinin\n PCA ile Projeksiyonu")

kernel_pca_proj_ax.scatter(X_test_kernel_pca[:, 0], X_test_kernel_pca[:, 1], c=y_test)
kernel_pca_proj_ax.set_ylabel("Temel Bileşen #1")
kernel_pca_proj_ax.set_xlabel("Temel Bileşen #0")
_ = kernel_pca_proj_ax.set_title("Test Verisinin\n KernelPCA ile Projeksiyonu")


# In[78]:


X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
X_reconstructed_kernel_pca = kernel_pca.inverse_transform(kernel_pca.transform(X_test))


# In[79]:


fig, (orig_data_ax, pca_back_proj_ax, kernel_pca_back_proj_ax) = plt.subplots(
    ncols=3, sharex=True, sharey=True, figsize=(13, 4)
)

orig_data_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
orig_data_ax.set_ylabel("Özitelik #1")
orig_data_ax.set_xlabel("Özitelik #0")
orig_data_ax.set_title("Orjinal Test Verisi")

pca_back_proj_ax.scatter(X_reconstructed_pca[:, 0], X_reconstructed_pca[:, 1], c=y_test)
pca_back_proj_ax.set_xlabel("Öznitelik #0")
pca_back_proj_ax.set_title("PCA ile Geri Oluşturulmuş Test Verisi")

kernel_pca_back_proj_ax.scatter(
    X_reconstructed_kernel_pca[:, 0], X_reconstructed_kernel_pca[:, 1], c=y_test
)
kernel_pca_back_proj_ax.set_xlabel("Özenitelik #0")
_ = kernel_pca_back_proj_ax.set_title("KernelPCA ile Geri Oluşturulmuş Test Verisi")


# In[81]:


from sklearn.feature_selection import VarianceThreshold
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
X


# In[82]:


sel = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))
sel.fit_transform(X)


# In[83]:


from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X, y = load_iris(return_X_y=True)
X.shape


# In[84]:


X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
X_new.shape


# In[86]:


from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
X, y = load_iris(return_X_y=True)
X.shape


# In[87]:


lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
X_new.shape


# In[88]:


lsvc = LinearSVC(C=0.01, penalty="l2", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
X_new.shape


# In[90]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
X, y = load_iris(return_X_y=True)
X.shape


# In[91]:


clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)
clf.feature_importances_


# In[92]:


model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X_new.shape


# In[93]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris


# In[94]:


iris = load_iris()

x = iris.data
y = iris.target


# In[95]:


x = (x-np.min(x))/(np.max(x)-np.min(x))


# In[96]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)


# In[97]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn


# In[98]:


from sklearn.model_selection import cross_val_score
fold_sayisi = 10
dogruluklar = cross_val_score(estimator = knn, X = x_train, y=y_train, cv = fold_sayisi)
print("Ortalama Doğruluk: ",np.mean(dogruluklar))
print("Doğrulukların Standart Sapması: ",np.std(dogruluklar))


# In[99]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris


# In[100]:


iris = load_iris()

x = iris.data
y = iris.target


# In[101]:


x = (x-np.min(x))/(np.max(x)-np.min(x))


# In[102]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)


# In[104]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)


# In[105]:


from sklearn.model_selection import GridSearchCV

grid = {"n_neighbors":np.arange(1,50)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv = 10)
knn_cv.fit(x,y)


# In[111]:


print("En iyi K değeri: ",knn_cv.best_params_)
print("En iyi K değerine göre en iyi doğruluk değeri: ",knn_cv.best_score_)


# In[114]:


from sklearn.linear_model import LogisticRegression

grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,grid,cv = 10)
logreg_cv.fit(x,y)

print("En iyi hiper parametreler: ",logreg_cv.best_params_)
print("En iyi hiper parametrelere göre en iyi doğruluk değeri: ",logreg_cv.best_score_)

