import xml.etree.ElementTree as ET
import numpy as np

label_file_path_1 = '../dataset/svt1/train.xml'
label_file_path_2 = '../dataset/svt1/test.xml'

def extract_bounding_boxes_from_xml(xml_file):
    labels = {}
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for image in root.findall('image'):
        name = image.find('imageName').text[4:9]
        for rects in image.findall('taggedRectangles'):
            tmp = [] 
            for rect in rects.findall('taggedRectangle'):

                #Estrai il label di ogni bounding box nell'immagine
                h= rect.get('height')
                w= rect.get('width')
                x= rect.get('x')
                y= rect.get('y')

                tmp.append([int(x),int(y),int(w),int(h), 0])

                #aggrega le bounding b della stessa immagine in un solo array
            #per ogni immagine aggiungi il label alla lista dei label
            labels[name] = (np.array(tmp))
    return labels

def extract_labels_from_xml(xml_file):
    labels = {}
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for image in root.findall('image'):
        name = image.find('imageName').text[4:9]
        labels[name] = []
        for rects in image.findall('taggedRectangles'):
            for rect in rects.findall('taggedRectangle'):
                tmp = {}
                rect_name = rect.find('tag').text
                #Estrai il label di ogni bounding box nell'immagine
                h= rect.get('height')
                w= rect.get('width')
                x= rect.get('x')
                y= rect.get('y')

                tmp['tag'] = rect_name
                tmp['box'] = ([int(x),int(y),int(w),int(h), 0])

                #aggrega le bounding b della stessa immagine in un solo array
                #per ogni immagine aggiungi il label alla lista dei label
                labels[name].append(tmp)
    return labels

def save_bounding_boxes(train, test):
    labels_1 = extract_bounding_boxes_from_xml(train)
    labels_2 = extract_bounding_boxes_from_xml(test)
    labels = {**labels_1, **labels_2}
    labels_arr = []
    for key, i in sorted(labels.items()):
        labels_arr.append(i)
    labels_arr = np.array(labels_arr)
    np.save("SVT_bounding_boxes", labels_arr)
    
    
save_bounding_boxes(label_file_path_1, label_file_path_2)
labels = extract_labels_from_xml(label_file_path_1)