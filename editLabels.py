import xml.etree.ElementTree as ET

file_path = 'Datasets/SVT/trainCopy.xml'
tree = ET.parse(file_path)
root = tree.getroot()

for image in root.findall('image'):
    for lex in image.findall('lex'):
        image.remove(lex)
    for address in image.findall('address'):
       image.remove(address)


tree.write(file_path)