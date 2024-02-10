import sys
import os
import csv
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms

mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]
test_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean_nums, std_nums)
])
test_set_path = sys.argv[1]
chosen_dataset = datasets.ImageFolder(test_set_path, test_transforms)
labels = chosen_dataset.classes
print(labels)
USE_CPU = os.getenv('USE_CPU')
if USE_CPU == "TRUE":
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
path_model = sys.argv[2]
model = torch.load(path_model)
model.eval()


def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
    top_percentage = torch.topk(percentage, 1)
    acc = top_percentage[0][0].item()
    label_index = top_percentage[1][0].item()
    return acc, label_index


csv_file_name = "individual_eval.csv"
csv_top_one_file_name = "top_one_accuracy.csv"
fieldnames = ['label', 'prediction_true', 'prediction_false', 'is_predicted']
top_one_fieldnames = ['image_name', 'ground_truth', 'prediction', 'accuracy']
if os.path.exists(csv_file_name):
    os.remove(csv_file_name)
    with open(csv_file_name, mode='a+') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
if os.path.exists(csv_top_one_file_name):
    os.remove(csv_top_one_file_name)
    with open(csv_top_one_file_name, mode='a+') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=top_one_fieldnames)
        writer.writeheader()


total_num_images = 0
prediction_list = []
prediction_rate = 0.6
i = 0
j = 0
k = 0
for ground_truth in os.listdir(test_set_path):
    prediction = {}
    prediction_true = 0
    prediction_false = 0
    is_predicted = False
    # jimmy  <= 100 in case we have class others 
    if j <= 100:
        for image in os.listdir(test_set_path + "/" + ground_truth):
            total_num_images = total_num_images + 1
            image_path = test_set_path + "/" + ground_truth + "/" + image
            im = Image.open(image_path).convert('RGB')
            acc, index = predict_image(im)
            predicted_class = labels[index]
            print('p', predicted_class)
            print('gt', ground_truth)
            print('acc', acc)
            with open(csv_top_one_file_name, mode='a+') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=top_one_fieldnames)
                writer.writerow(
                    {top_one_fieldnames[0]: image, top_one_fieldnames[1]: ground_truth,
                     top_one_fieldnames[2]: predicted_class, top_one_fieldnames[3]: acc})
            if predicted_class == ground_truth:
                prediction_true = prediction_true + 1
                i = i + 1
            else:
                prediction_false = prediction_false + 1
        if prediction_true / (prediction_true + prediction_false) >= prediction_rate:
            is_predicted = True
            k = k + 1
        prediction = {"label": ground_truth, "prediction_true": prediction_true, "prediction_false": prediction_false,
                      "is_predicted": is_predicted}
        prediction_list.append(prediction)
        with open(csv_file_name, mode='a+') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow(
                {fieldnames[0]: ground_truth, fieldnames[1]: prediction_true,
                 fieldnames[2]: prediction_false, fieldnames[3]: is_predicted})
    j = j + 1
print(prediction_list)
print("total is_predicted", k)
print("total images", total_num_images)
print("total is_match", i)
print(path_model)
