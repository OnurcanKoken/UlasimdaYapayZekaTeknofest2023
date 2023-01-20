import logging
import time

import requests

from src.detected_object import DetectedObject
from utils.detect import Yolo


class ObjectDetectionModel(Yolo):
    # Base class for team models
    def __init__(self, evaluation_server_url):
        Yolo.__init__(self)
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url
        # Modelinizi bu kısımda init edebilirsiniz.
        # self.model = get_keras_model() # Örnektir!

    @staticmethod
    def download_image(img_url, images_folder, prediction):
        t1 = time.perf_counter()
        img_bytes = requests.get(img_url).content
        image_name = img_url.split("/")[-1]  # frame_x.jpg

        with open(images_folder + image_name, 'wb') as img_file:
            img_file.write(img_bytes)

        t2 = time.perf_counter()

        prediction.frame_path = images_folder + image_name
        logging.info(f'{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}')

    def process(self, prediction, evaluation_server_url):
        # Yarışmacılar resim indirme, pre ve post process vb işlemlerini burada gerçekleştirebilir.
        # Download image (Example)
        self.download_image(evaluation_server_url + "media" + prediction.image_url, "./_images/", prediction)
        # Örnek: Burada OpenCV gibi bir tool ile preprocessing işlemi yapılabilir. (Tercihe Bağlı)
        # ...
        # Nesne tespiti modelinin bulunduğu fonksiyonun (self.detect() ) çağırılması burada olmalıdır.
        frame_results = self.detect(prediction)
        # Tahminler objesi FramePrediction sınıfında return olarak dönülmelidir.
        return frame_results

    def detect(self, prediction):
        # Modelinizle bu fonksiyon içerisinde tahmin yapınız.
        detect_objets, landing = self.detect_model(prediction.frame_path)
        # Burada örnek olması amacıyla 20 adet tahmin yapıldığı simüle edilmiştir.
        # Yarışma esnasında modelin tahmin olarak ürettiği sonuçlar kullanılmalıdır.
        # Örneğin :
        # for i in results: # gibi
        for i, object in enumerate(detect_objets):
            cls = int(object[5])
            landing_status = landing[i]
            top_left_x = object[0]
            top_left_y = object[1]
            bottom_right_x = object[2]
            bottom_right_y = object[3]

            # Modelin tespit ettiği herbir nesne için bir DetectedObject sınıfına ait nesne oluşturularak
            # tahmin modelinin sonuçları parametre olarak verilmelidir.
            d_obj = DetectedObject(cls,
                                   landing_status,
                                   top_left_x,
                                   top_left_y,
                                   bottom_right_x,
                                   bottom_right_y)
            # Modelin tahmin ettiği her nesne prediction nesnesi içerisinde bulunan detected_objects listesine eklenmelidir.
            prediction.add_detected_object(d_obj)

        return prediction
