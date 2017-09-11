import xml.etree.ElementTree
from xml.dom import minidom

def generateFileIDs(dataset_class_path):
    with open(dataset_class_path, "r") as f:
        class_of_interest_ids = []
        all_lines = f.readlines()
        for line in all_lines:
            try:
                fileid, fileclass = line.strip().split(" ")
            except ValueError:
                # print line
                fileid, garbage, fileclass = line.strip().split(" ")
            if(int(fileclass) == 1):
                class_of_interest_ids.append(fileid)
    print("Found " + str(len(class_of_interest_ids)) + " files in this class.")
    return class_of_interest_ids

def countObjects(file_ids, dataset_annotations_path):
    filtered_ids = []
    for anno_file in file_ids:
        e = xml.etree.ElementTree.parse(dataset_annotations_path + anno_file + ".xml").getroot()
        xml_anno = minidom.parse(dataset_annotations_path + anno_file + ".xml")
        num_objects = xml_anno.getElementsByTagName('object')
        # print(anno_file + " : " + str(len(num_objects)))
        if(len(num_objects) == 1):
            filtered_ids.append(anno_file)
    print "Number of images with only class of interest : " + str(len(filtered_ids))
    return filtered_ids


dataset_class_path = "VOC2007/ImageSets/Main/cat_val.txt"
dataset_annotations_path = "VOC2007/Annotations/"
print countObjects(generateFileIDs(dataset_class_path), dataset_annotations_path)
