import os
import shutil
import sys
import shutil

original_dataset = sys.argv[1]
destination_path = sys.argv[2]
number_of_clients = sys.argv[3]
number_of_classes_per_client = sys.argv[4]
number_of_images_per_class = sys.argv[5]
training_ratio = sys.argv[6]
validation_ratio = sys.argv[7]
num_test_images = sys.argv[8]

folders = []
used_images = []
os.makedirs(os.path.join(destination_path, 'test'))
test_imgs = 0

for i in range(int(number_of_clients)):
    path = os.path.join(destination_path, str(i+1))
    os.makedirs(path)
    os.makedirs(path+'/train')
    os.makedirs(path+'/val')
    classes = 0
    val_imgs = 0
    train_imgs = 0
    for folder_name in os.listdir(original_dataset):
        p = os.path.join(original_dataset, folder_name)
        os.makedirs(path+'/train/'+folder_name)
        os.makedirs(path+'/val/'+folder_name)
        train_image_num = int(int(number_of_images_per_class) * int(training_ratio) / 100) 
        val_image_num = int(int(number_of_images_per_class) * int(validation_ratio) / 100)
        for image_name in os.listdir(original_dataset+"/"+folder_name):
            if image_name not in used_images:
                train_imgs = train_imgs + 1
                shutil.copyfile(original_dataset+"/"+folder_name+"/"+image_name, destination_path+str(i+1)+"/train/"+folder_name+"/"+image_name)
                used_images.append(image_name)
            if(train_imgs == int(train_image_num)):
                break
        train_imgs = 0
        for image_name in os.listdir(original_dataset+"/"+folder_name):
            if image_name not in used_images:
                val_imgs = val_imgs + 1
                shutil.copyfile(original_dataset+"/"+folder_name+"/"+image_name, destination_path+str(i+1)+"/val/"+folder_name+"/"+image_name)
                used_images.append(image_name)
            if(val_imgs == int(val_image_num)):
                break
        val_imgs = 0
        classes = classes + 1
        if(classes == int(number_of_classes_per_client)):
            break

classes = 0
for folder_name in os.listdir(original_dataset):
    os.makedirs(os.path.join(destination_path, "test/"+folder_name))
    for image_name in os.listdir(original_dataset+"/"+folder_name):
        if image_name not in used_images:
            test_imgs = test_imgs + 1
            shutil.copyfile(original_dataset+"/"+folder_name+"/"+image_name, destination_path+"test/"+folder_name+"/"+image_name)
            used_images.append(image_name)
        if(test_imgs == int(num_test_images)):
            break
    test_imgs = 0
    classes = classes + 1
    if(classes == int(number_of_classes_per_client)):
        break
