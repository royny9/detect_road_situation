import pandas as pd
import matplotlib.pyplot as plt
import os



dir_labels = os.listdir('bdd_new/labels/train')
# print(type(dir_labels))


hash_cls_card = dict()
tw = 0
for i in  dir_labels:
    with open(f'bdd_new/labels/train/{i}', 'r') as file:
        s = file.readlines()
        if len(s) >0:
            cls_idx = s[0][0]
            if cls_idx in hash_cls_card:
               hash_cls_card[cls_idx] += 1
            else:
                hash_cls_card[cls_idx] = 1
            
print(hash_cls_card)



plt.bar(hash_cls_card.keys(), hash_cls_card.values(), color='skyblue')

plt.xlabel('cls')
plt.ylabel('value')
plt.show()


'''очень сильный дисбаланс класса 1'''