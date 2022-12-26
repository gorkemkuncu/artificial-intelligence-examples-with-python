#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[5]:


# Outcome = 1 Diabet Hastası
# Outcome = 0 Sağlıklı

data = pd.read_csv("diabetes.csv")
data.head()


# In[15]:


seker_Hastalari = data[data.Outcome == 1]
saglikli_Insanlar = data[data.Outcome == 0]

# Şimdilik gloucose'yi referans alarak bir örnek yapalım
# Programın sonunda makine öğrenme modeli sadece glikoza değil tüm değerlere bakarak tahmin yapacaktır

plt.scatter(saglikli_Insanlar.Age, saglikli_Insanlar.Glucose, color="green", label="sağlıklı", alpha=0.4)
plt.scatter(seker_Hastalari.Age, seker_Hastalari.Glucose, color="red",label= "diabet hastası", alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend()
plt.show()


# In[26]:


# x ve y eksenlerini belirtelim
y = data.Outcome.values
x_Ham_Veri = data.drop(["Outcome"],axis=1)
# Outcome sütununu(dependent variable) çıkarıp sadece independent variables bırakıyoruz
# Çünkü KNN algoritması x değerleri içerisinde gruplandırma yapacak


# Normalizasyon
x = (x_Ham_Veri - np.min(x_Ham_Veri)) / (np.max(x_Ham_Veri) - np.min(x_Ham_Veri))

# Normalizasyon öncesi ham veriler
print("Normalizasyon öncesi ham veriler: \n")
print(x_Ham_Veri.head())

# Normalizasyon sonrası ham veriler
print("\n\n\nNormalizasyon sonrası ham veriler: \n")
print(x.head())


# In[42]:


# test ve train datalarını ayırıyouz
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

# KNN modelini oluşturuyoruz
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("K=3 için test değerlerimizin doğrulama sonucu", knn.score(x_test, y_test))


# In[43]:


# k kaç olmalı 
# en iyi k değerini bulalım

sayac = 1
for k in range(1,11):
    knn_yeni = KNeighborsClassifier(n_neighbors=k)
    knn_yeni.fit(x_train,y_train)
    print(sayac," ","Doğruluk oranı: %", knn_yeni.score(x_test,y_test)*100)
    sayac += 1 


# In[ ]:




