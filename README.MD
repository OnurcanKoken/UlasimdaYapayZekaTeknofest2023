# Artificial Intelligence in Transportation Competition

In this competition, participants will be expected to detect objects using pre-recorded footage from a UAV (Unmanned Aerial Vehicle). The UAV will represent footage from an aircraft flying in low-altitude including flight maneuvers and images in focus. The footage that will be used for testing purposes during the competition will have been labeled beforehand, and the competitors will be scored for their accuracy of detection within the given time limit. These objects will be of detectable quality as they will be variations of objects available in the footage broadcast during the competition, sample videos and training videos. It should be considered that flying cars serve at night, during the day and in harsh weather conditions
(rain, fog, snow)

Tasks
The competitors will detect objects using pre-recorded footage from a UAV (Unmanned Aerial Vehicle). 

Types of Objects Subject to Detection
The objects will be of detectable quality as they will be variations of objects available in the footage broadcast during the competition, sample videos and training videos. Types of objects subject to detection include:
-    Vehicles (tasit) - class #0
-    Humans (insan) - class #1
-    Flying Car Parking (FCP/UAP) Area - class #2
-    Landing Attitude of Flying Car Parking
-    Flying Ambulance Landing (FAL/UAI) Area - class #3
-    Landing Attitude of Flying Ambulance Landing

# Dataset
The dataset we have used is a custom dataset. Collected from previous competitions of Teknofest, people-on-grass, Visdrone. Total: 16606 labeled images are used.

# Model
YOLOv7.

# Preprocessing
Working on dehazing for foggy cases. It works nice but detection of the existance of foggy weather is a bit problem.

# Yarışmacılar için Örnek Sunucu Bağlantı Arayüzü
Bu repository Teknofest Ulaşımda Yapay Zeka yarışması değerlendirme sunucusuna(TUYZDS) örnek bağlantının kurulması için geliştirilmiştir.
Yarışmacılar örnek arayüzü ihtiyaçlarına göre güncelleyebilir yada farklı bir dil ile (C++, vs) benzerini implemente edebilirler.
 

# Kurulum
- Sistem için bir sanal ortam oluşturulur. (Anaconda'nın kurulu olduğu varsayılmaktadır.)
```shell
$ conda create -n teknofest_yarisma python=3.7
$ conda activate teknofest_yarisma
```
- Gerekli paketler yüklenir.
```shell
$ pip install -r requirements.txt
```
- `./config` klasorü altında bulunan `example.env` klasörü referans alınarak (kopyalanarak) `.env` isiminde bir dosya oluşturulur.
- `.env` dosyasının içeriği teknofest tarafından verilen takım kullanıcı adı ve şifreleri ile güncellenerek kaydedilir. (Dikkat: Sonuçta oluşacak dosyanın yolu `./config/.env` şeklinde olacaktır.) 
````text
TEAM_NAME=takim_adi
PASSWORD=sifre
EVALUATION_SERVER_URL=http://teknofest.cezerirobot.com:2052/
````
- Kod içerisinde yorumlar dikkate alınarak gerekli düzenlemeler ve geliştirmeler yapılır. 
  ```
  NOT: 
  İlk testinizde sadece `.env` dosyasında değişiklik yaparak sunucu iletişiminizi test edebilirsiniz. 
  ```
- Sistemi çalıştırmak için aşağıdaki komut çalıştırılır.
````shell
python main.py
````
# Yarışmacıların Modellerini Tanımlayacakları Kısımlar
Yarışmacılar sağlanan sunucu haberleşme arayüzüne ait kodları modifiye etmeden, yanlızca ```object_detection_model.py``` içerisinde değişiklik yaparak modellerini sisteme entegre edebilirler.
Geliştirilen modellerin ``ObjectDetectionModel`` sınıfına entegrasyonu ile ilgili detaylar kod içerisinde yorumlar ile belirtilmiştir.

## Logların İncelenmesi
Sistem çalışma esnasında gerçekleştirdiği işlemleri `_logs` klasörü altında kayıt altına almaktadır. Sistem hataları bu log dosyaları üzerinden takip edilebilir. Yarışma esnasında yapılacak itirazlarda log dosyası içeriği değerlendirmeye alınacaktır.

# Sorumluluk Beyanı
Bu sistem yarışmacılar için referans olarak sunulmuştur. Bu sistemi kullanmak kullanıcıların tercihine bağlıdır. Sağlanan sistem dolayısı ile oluşabilecek problemler yarışmacıların sorumluluğundadır.
Yarışmacı takımlar sunucu ile sağlıklı iletişim kuracak yapıyı geliştirmek/test etmek sorumluluğundadır. Teknofest, sistemde değişiklik yapma hakkını saklı tutar.