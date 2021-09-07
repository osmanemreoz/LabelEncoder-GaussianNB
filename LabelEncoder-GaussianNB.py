import pandas as pd
import numpy as np
import math

a1 = pd.DataFrame(["Evet","Hayır","Evet","Hayır","Evet","Hayır","Evet","Hayır","Hayır","Hayır"], columns = ["Kaza"]) #sütun ismini Kaza yazdık
a2 = pd.DataFrame([25,18,39,50,45,22,35,28,22,40], columns = ["Yaş"]) #sütun ismini Yaş yazdık
a3 = pd.DataFrame(["Erkek","Kadın","Kadın","Kadın","Erkek","Erkek","Erkek","Erkek","Kadın","Erkek"], columns = ["Cinsiyet"]) #sütun ismini Cinsiyet yazdık
#a4 = pd.DataFrame(["Yüksek","Yüksek","Orta","Düşük","Düşük","Orta","Orta","Orta","Orta","Yüksek"], columns = ["Risk"]) #sütun ismini Risk yazdık.
a4 = pd.DataFrame([2,2,1,0,0,1,1,1,1,0])
#burda Yüksek Risk:2 , Orta Risk:1 , Düşük Risk:0 olarak tanımladık



m1 = pd.concat([a1,a2,a3], axis = 1) #axis=1 yani sütunsal olarak birleştirdik

#Kaza sütunu ve Cinsiyet Sütunu kategorik. nümerik yapmamız lazım.

from sklearn.preprocessing import LabelEncoder #Kategorikten Nümerik'e çevirmek için bu kütüphaneyi importladık.
lb = LabelEncoder()

for i in range(0, 1): #burda döngü kurduk. for i in range(0,1) yani i 0dan 1e kadar. range(0,1) demek aslında 0'ı al demektir. yani sağ ucu almıyoruz.
    m1.iloc[:,i:i+1] = lb.fit_transform(m1.iloc[:,i:i+1]) #m1'in içinden tüm satırları al , i'den i+1'e kadar . yani m1'in içinden 0'dan 1.sütuna kadar olan sütunu. yani aslında kaza sütununu alıcak.
#lb.fit_transform(m1.iloc[:,i:i+1]) yani kaza sütununa labelencoder'ı uyguladık burda ve bu uyguladığımız değerleri oraya yapıştırdık.


for i in range(2, 3): #for i in range(2,3) yani i 2'den 3'e kadar. range(2,3) demek aslında 2yi al demektir.
    m1.iloc[:,i:i+1] = lb.fit_transform(m1.iloc[:,i:i+1])  
#burda da cinsiyet sütununa labelencoder'ı uyguladık ve bu uyguladığımız değerleri yapıştırdık.

#NOT: m1 bizim girdilerimiz!


from sklearn.naive_bayes import GaussianNB #içerde yani girdide nümerik varsa ona gauss dağılımını uygular bu kütüphane.
gnb = GaussianNB() #gnb değişkenine atadık.
gnb.fit(m1, a4) #modeli eğittik burda.

y_pred = gnb.predict(np.array([1,42,1]).reshape(1,3)) #[1,42,1] demek kaza yapmamış 42 yaşında bir kadının riskini yukardaki modele göre yani naive_bayes'e göre tahmin et. reshape(1,3) 1 satır 3 sütun demek yan yana koymak demek.             
print(y_pred)
print(gnb.predict_proba(np.array([1,42,1]).reshape(1,3)))

#predict_proba demek olasılıklarını göstermek demek.



#NOT: a4'ü kategorik alırsak console'da Orta Riskli çıkıyor. Nümerik alırsak Düşük Riskli çıkıyor.
#Nümerik olarak aldığımız için LabelEncoder uyguladık.


## Üstteki yaptıklarımızın manuel ##
# c1=[45,50]
# ort1=np.mean(c1)
# sapma1=np.std(c1)
# sapma1=sapma1*np.sqrt(len(c1)/(len(c1)-1))
# c2=[39,22,35,28,22]
# ort2=np.mean(c2)
# sapma2=np.std(c2)
# sapma2=sapma2*np.sqrt(len(c2)/(len(c2)-1))
# c3=[25,18,40]
# ort3=np.mean(c3)
# sapma3=np.std(c3)
# sapma3=sapma3*np.sqrt(len(c3)/(len(c3)-1))

# g1=(1/(sapma1*np.sqrt(2*math.pi))*math.exp(-0.5*(42-ort1)/sapma1)**2)
# g2=(1/(sapma2*np.sqrt(2*math.pi))*math.exp(-0.5*(42-ort1)/sapma2)**2)
# g3=(1/(sapma3*np.sqrt(2*math.pi))*math.exp(-0.5*(42-ort1)/sapma3)**2)

# g1=g1*0.5*0.5*0.2
# g2=g2*0.6*0.2*0.5
# g3=g3*(2/3)*(1/3)*0.3

# S=g1+g2+g3

# P1=g1/S
# P2=g2/S
# P3=g3/S















