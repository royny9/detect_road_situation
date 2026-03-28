import json
import os
from PIL import Image

# class toYoloFormat():
#     def __init__(self, path_labels, path_img=None):
#         self.new_dict = dict()
#         self.obj_nn = ['car', 'pothole', 'person']
#         with open(path_labels, 'r') as labels:
#             json_labels = json.load(labels)
#             for item in json_labels:
#                 labels_on_img = item['labels']
#                 cls_to_box = dict()
#                 for cat in labels_on_img:
#                     if cat['category'] in self.obj_nn:
#                         cls_to_box[cat['category']] = cat['box2d']



class toYoloFormat():
    def __init__(self, path_labels, path_img_folder, mode=None):
        
        self.obj_nn = {
            'car':0,
            'person':1,
            'pothole':2
        }
        self.path_labels = path_labels
        self.mode = mode
        self.labels_dir = f'bdd_new/labels/{self.mode}'
        self.dir_folder_labels = os.makedirs(self.labels_dir, exist_ok=True)
        self.path_img_folder = path_img_folder


    def crate_txt_yolo(self):
        with open(self.path_labels) as l:
            label_inp = json.load(l)
            for item in label_inp:
                labels = item['labels']
                #чтение изображения
                if os.path.exists(f"{self.path_img_folder}/{item['name']}"):
                    img = Image.open(f"{self.path_img_folder}/{item['name']}")
                else:
                    print('не удалось найти изображение ')
                    continue
                width, height = img.size
                #обработка значений
                with open(f'{self.labels_dir}/{item['name'].split('.')[0]}.txt', 'w') as txt:
                    for element in labels:
                        if element['category'] in self.obj_nn:
                            x1 = element['box2d']['x1']
                            x2 = element['box2d']['x2']
                            y1 = element['box2d']['y1']
                            y2 = element['box2d']['y2']
                            if x1>=x2 or y1>y2:
                                continue
                            x_center = max(0,min(1,((x1+x2) / 2 / width)))
                            y_center = max(0,min(1,((y1+y2) / 2 / height)))
                            box_w = max(0,min(1,((x2-x1) / width)))
                            box_h = max(0,min(1,((y2-y1) / height)))
                            name_obj = self.obj_nn[element['category']]
                            txt.write(f'{name_obj} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n')
                        

# {
#         "name": "0000f77c-6257be58.jpg",
#         "attributes": {
#             "weather": "clear",
#             "scene": "city street",
#             "timeofday": "daytime"
#         },
#         "timestamp": 10000,
#         "labels": [
#             {
#                 "category": "traffic light",
#                 "attributes": {
#                     "occluded": false,
#                     "truncated": false,
#                     "trafficLightColor": "green"
#                 },
#                 "manualShape": true,
#                 "manualAttributes": true,
#                 "box2d": {
#                     "x1": 1125.902264,
#                     "y1": 133.184488,
#                     "x2": 1156.978645,
#                     "y2": 210.875445
#                 },
#                 "id": 0
#             },
#             {
#                 "category": "traffic light",
#                 "attributes": {
#                     "occluded": false,
#                     "truncated": false,
#                     "trafficLightColor": "green"
#                 },
#                 "manualShape": true,
#                 "manualAttributes": true,
#                 "box2d": {
#                     "x1": 1156.978645,
#                     "y1": 136.637417,
#                     "x2": 1191.50796,
#                     "y2": 210.875443
#                 },
#                 "id": 1
#             },                


a = toYoloFormat(path_labels='bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json',
                 mode='val',
                 path_img_folder='bdd/bdd100k/bdd100k/images/100k/val')

a.crate_txt_yolo()