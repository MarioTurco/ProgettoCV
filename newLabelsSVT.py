import xml.etree.ElementTree as ET
import numpy as np

label_file_path = 'Datasets/SVT/trainCopy.xml'
number_of_images = 350


labels = []


tree = ET.parse(label_file_path)
root = tree.getroot()

for image in root.findall('image'):
    for rects in image.findall('taggedRectangles'):
        tmp = [] 
        for rect in rects.findall('taggedRectangle'):

            #Estrai il label di ogni bounding box nell'immagine
            h= rect.get('height')
            w= rect.get('width')
            x= rect.get('x')
            y= rect.get('y')

            tmp.append([h,w,x,y])
            
            #aggrega le bounding b della stessa immagine in un solo array
        #per ogni immagine aggiungi il label alla lista dei label
        labels.append(np.array(tmp))
        
labels = np.array(labels)
np.save("SVT_labels", labels=labels, allow_pickle=True)
# print(labels[1])
# print(labels)
