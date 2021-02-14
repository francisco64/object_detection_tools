import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_pandas(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bbx = member.find('bndbox')
            xmin = int(bbx.find('xmin').text)
            ymin = int(bbx.find('ymin').text)
            xmax = int(bbx.find('xmax').text)
            ymax = int(bbx.find('ymax').text)
            label = member.find('name').text

            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     label,
                     xmin,
                     ymin,
                     xmax,
                     ymax
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'Class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def xml2pd(image_path,csv=False):
    xml_df = xml_to_pandas(image_path)
    if csv:
        xml_df.to_csv('dataset.csv', index=None)
        print('Successfully converted xml to csv. and saved to dataset.csv')
    return xml_df

# image_path = './fasterR-CNN/BCCD_Dataset-master/BCCD/Annotations'
# df=xml2pd(image_path)


 