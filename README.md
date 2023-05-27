# Predict-Advanced-House-Price---TF-Keras-Pycaret
DEEP LEARNING REGRESSION - PREDICT HOUSE NOT: TEST DOSYASINI BİR ÇOK YÖNTEM DENEMDİM. MODELE TAHMİN ETTİREMEDİM. İNVALİD ARGUMENT HATASI VERİYOR. FAKAT KENDİ MODELİNDE R2 SCORE 0,98
...................................libs.........................

        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense,Activation
        from tensorflow.keras.optimizers import Adam
..................................................................

          2 tane paketimiz var
          from tensorflow.keras.models import Sequential

          ve

          from tensorflow.keras.layers import Dense

          Data Dictionary
          SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
          MSSubClass: The building class
          MSZoning: The general zoning classification
          LotFrontage: Linear feet of street connected to property
          LotArea: Lot size in square feet
          Street: Type of road access
          Alley: Type of alley access
          LotShape: General shape of property
          LandContour: Flatness of the property
          Utilities: Type of utilities available
          LotConfig: Lot configuration
          LandSlope: Slope of property
          Neighborhood: Physical locations within Ames city limits
          Condition1: Proximity to main road or railroad
          Condition2: Proximity to main road or railroad (if a second is present)
          BldgType: Type of dwelling
          HouseStyle: Style of dwelling
          OverallQual: Overall material and finish quality
          OverallCond: Overall condition rating
          YearBuilt: Original construction date
          YearRemodAdd: Remodel date
          RoofStyle: Type of roof
          RoofMatl: Roof material
          Exterior1st: Exterior covering on house
          Exterior2nd: Exterior covering on house (if more than one material)
          MasVnrType: Masonry veneer type
          MasVnrArea: Masonry veneer area in square feet
          ExterQual: Exterior material quality
          ExterCond: Present condition of the material on the exterior
          Foundation: Type of foundation
          BsmtQual: Height of the basement
          BsmtCond: General condition of the basement
          BsmtExposure: Walkout or garden level basement walls
          BsmtFinType1: Quality of basement finished area
          BsmtFinSF1: Type 1 finished square feet
          BsmtFinType2: Quality of second finished area (if present)
          BsmtFinSF2: Type 2 finished square feet
          BsmtUnfSF: Unfinished square feet of basement area
          TotalBsmtSF: Total square feet of basement area
          Heating: Type of heating
          HeatingQC: Heating quality and condition
          CentralAir: Central air conditioning
          Electrical: Electrical system
          1stFlrSF: First Floor square feet
          2ndFlrSF: Second floor square feet
          LowQualFinSF: Low quality finished square feet (all floors)
          GrLivArea: Above grade (ground) living area square feet
          BsmtFullBath: Basement full bathrooms
          BsmtHalfBath: Basement half bathrooms
          FullBath: Full bathrooms above grade
          HalfBath: Half baths above grade
          Bedroom: Number of bedrooms above basement level
          Kitchen: Number of kitchens
          KitchenQual: Kitchen quality
          TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
          Functional: Home functionality rating
          Fireplaces: Number of fireplaces
          FireplaceQu: Fireplace quality
          GarageType: Garage location
          GarageYrBlt: Year garage was built
          GarageFinish: Interior finish of the garage
          GarageCars: Size of garage in car capacity
          GarageArea: Size of garage in square feet
          GarageQual: Garage quality
          GarageCond: Garage condition
          PavedDrive: Paved driveway
          WoodDeckSF: Wood deck area in square feet
          OpenPorchSF: Open porch area in square feet
          EnclosedPorch: Enclosed porch area in square feet
          3SsnPorch: Three season porch area in square feet
          ScreenPorch: Screen porch area in square feet
          PoolArea: Pool area in square feet
          PoolQC: Pool quality
          Fence: Fence quality
          MiscFeature: Miscellaneous feature not covered in other categories
          MiscVal: $Value of miscellaneous feature
          MoSold: Month Sold
          YrSold: Year Sold
          SaleType: Type of sale
          SaleCondition: Condition of sale
          
          
          BU YÖNTEM BOŞ SÜTUNLARI ELEMEK İÇİN ÖNEMLİ %20 DEN FAZLASI BOŞ OLANI DÜŞÜR KOMUTU
DROPNA DAKİ TRESH FONKSİYONU= 2 İSE AXİS 0 DA YANİ SATIRDA 2 TANE NULL VARSA ELİYOR. AXİS 1 İSE SÜTUNDA 2 TANE VARSA. FAKAT int(0.8*len(df)) YAZILDIĞINDA TOPLAMDA %80 ANLAMINDA OLUYOR BU ÇOK GÜZEL
ŞİMDİ DERİN ÖĞRENMEDE GİRİŞ LAYER I AÇMAK
LAYER EKLEME MODEL.ADD(DENSE,,,) İLE YAPILIYOR

Densi için 8 yazmamızın sebebi giriş sütununda 8 tane giriş var. bukadar nöron aç

İkinci layer 12 yapıldı, bir şeye bağlı olarak 12 yazılmadı nöron sayısını nekadar artıtırsanız o kadar başarı artıyor fakar bir noktandan sonra plato da kalıyor yani başarı yükselmiyor. orada durmak gerekiyor. tahmini olarak 12 yazıldı

Üçüncü 4 tahmini girildi.

FAKAT SONUÇ OLARAK 1 ÇIKACAĞI İÇİN SON LAYER 1 OLMASI LAZIM DİYABET HASTANSI OLACAK MI OLMAYACAKM MI

RELU da gelen veri önemli değilse 0 eşitliyor. Önemliyse ağırlığını yükseltiyor. Gelenler rakamsa Relu işler . Bunu araştırabilirsin

Sigmoid fonksiyonu  0 - 1 çıkartıyor classification da kullanılıyor.
Modeli tanımlıyoruz
Modeli model.compile ile tanımlıyoruz.

Loss demek modeli neye göre çalıştırmak istiyoruz, classification mı yapsın, regression mı, görüntü mü işlesin gibi

Loss= 'binary_crossentropy' diyerek yani classification yapmak için çalış cevabın 1 yada 0 olacağını söylüyoruz.

En iyi optimizer 'Adam' , metrics olarak classficationda 'accuracy' kullanıyorduk. fonksiyonun en derin noktasını buluyor. en kısa yoldan nasıl çözülür GRADİANT DESCENT grafiklerine bak. Eşşeğin dağa çıkarken en kolay yolu seçmesi gibi 
Metric olarak Accuracy yada MSE YADA RMS yazabilirsin
###

Verbose aşağıdaki rakamları göstermede kullanılıyor 1 olursa gösterir bu sayede çalışıtığını anlaşşıyor
Batch_size veri setinden 10 ar tane alarak işle
Epochs = sokaktan kaç defa geçeceğini söylüyoruz bunu çoğaltmak başarıyı yükseltir ama bir süre sonra başarı aynı kalmaya devam eder.
###

epochs u 150 yaptım accuracy= 79 oldu 500 yaptım 80 oldu.yani plato %80 başarı olabilir bunu layerleri çoğaltarak falan yükseltebilirisn
###

Markov zincirleri nedir bir bak
Yapılan adımlarıdaki değişimi bir history de tutabiliyorsun bunu da grafik olarak çizebiliyorsun
Algoritmanın age ve weight e aynı önem vermesini istiyoruz.
ÇOK ÖNEMLİ - HEDEFİ NORMALİZED EDEMESSİN ÇÜNKÜ FİYAT TAHMİNİ YAPAN MODEL FİYATI 0-1 ARASI TAHMİN EDER
Preprocessing demek datayı hazırlama
Tüm columns lardaki unique lere bakıyoru




from sklearn.preprocessing import normalize,scale

normaldata=scale(x)

normaldata.shape

sns.distplot(normaldata)

NORMALİZED ETMENİN DİĞER AVANTAJI KÜÇÜK RAKAMLAR İLE ÇALIŞILDIĞI İÇİN MAKİNEYİ YORMADAN ÇALIŞIYOR
normaldata.max(),normaldata.min()

REGRESSİON MODELİ ÇALIŞMA
ALGORİTHAYA HAZIR HALE GETİRDĞİMİZ DOSYAYI YENİDEN İHTİYACIMIZ ODLUĞUNDA ÇALIŞTIRABİLMEK İÇİN PİCKLE OLARKA KAYDETTİK
PİCKLE YANİ HAZIRLIĞI YAPILMIŞ DOSYAYI KAYDETME ŞEKLİ. İHTİYAÇ DURUMUNDA HAZIR DOSYAYI KULLAN
SADECE PİCKLE DOSYASINI ÇAĞIRARAK FİT PREDİCT OLARAK DEVAM EDEBİLİRSİN
Creating a Neural Network Model
Regression yaparken modeli tanımlarken loss = 'mse' yapmak gerekiyor.
LSTM Long Short Term Memory bu resim üretebiliyor. resim çizebiliyor, roman yazabiliyor. Sadce aşağıdak cnc2D
model=Sequential() model.add(tf.keras.layers.InputLayer(input_shape=(28,28))) model.add(tf.keras.layers.Reshape(target_shape=(28,28,1)) )

model.add(tf.keras.layers.Conv2D DEĞİL LSTN (filters=12,kernel_size=(3,3),activation='relu')) # BURADA CONV2D YERİNDE LSTN

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2))) model.add(tf.keras.layers.Flatten()) model.add(tf.keras.layers.Dense(10))


![image](https://github.com/omrbhdr/Predict-Advanced-House-Price---TF-Keras-Pycaret/assets/12261537/61103469-0745-48c7-b53d-f7712898fb47)



PYCARET

![image](https://github.com/omrbhdr/Predict-Advanced-House-Price---TF-Keras-Pycaret/assets/12261537/b031f28e-07c7-42f4-91e2-71dd93f638fd)

![image](https://github.com/omrbhdr/Predict-Advanced-House-Price---TF-Keras-Pycaret/assets/12261537/813575bc-d22c-4b61-8821-70233be4387e)
