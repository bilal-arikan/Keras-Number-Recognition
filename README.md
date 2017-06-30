# Keras-Number-Recognition

Kaynaklar:
http://erdoganb.com/2017/04/kerastensorflow-ile-rakamlari-tanima-mnist-dataset/
https://medium.com/turkce/keras-ile-derin-%C3%B6%C4%9Frenmeye-giri%C5%9F-40e13c249ea8

Kullanımı:

Eğitim için olan kısmı parametresiz çalıştırmanız yeterli, Mnist verisini internetten alıp, alınan veriye renkleri ters çevrilmiş hallerininde ekleyip (Doğruluk oranı ciddi miktarda arttı) eğitiyor
Var olan modeli kullanma ise, UsingModel.py ye bir resim konumunu parametre olarak gönderin çıktı olarak ağırlık değerlerini verecektir.

Dikkat !!!
Tahmin için kullancağınız resim Tek Kanallı (Siyah-Beyaz) ve 28 x 28 boyutunda olmalıdır 
