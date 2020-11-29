import glob, os
# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
current_dir = '/home/ankit/Downloads/gitWork/NFPA_dataset/NFPA dataset'
# Percentage of images to be used for the test set
percentage_test = 10;
# Create and/or truncate train.txt and test.txt
file_train = open('/home/ankit/Downloads/gitWork/Image_processing_Learning/darknet_master/backup/train.txt', 'w')
file_test = open('/home/ankit/Downloads/gitWork/Image_processing_Learning/darknet_master/backup/test.txt', 'w')
# Populate train.txt and test.txt
counter = 1
index_test = round(100 / percentage_test)
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
    print(pathAndFilename)
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    if counter == index_test:
        counter = 1
        print(title)
        file_test.write(current_dir + "/" + title + '.jpg' + "\n")
    else:
        file_train.write(current_dir + "/" + title + '.jpg' + "\n")
        counter = counter + 1
