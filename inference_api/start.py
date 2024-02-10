from fastapi import FastAPI, Form, File, UploadFile, Header, Body
import torch
from torchvision import datasets, models, transforms
from PIL import Image
from torch.autograd import Variable
import json
import os

app = FastAPI(version="1.0", title="torch classification API", description="API used for classification tasks using pytorch")
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]
test_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean_nums, std_nums)
])
USE_CPU = os.getenv('USE_CPU')
if USE_CPU == "TRUE":
    device = torch.device("cpu")
else:
    device = torch.device("cuda")

chosen_dataset = datasets.ImageFolder('/dataset/test',
                                      test_transforms)
labels = chosen_dataset.classes
model_dict = {}
for model_name in os.listdir('./models'):
    files = os.listdir('./models/'+model_name)
    m = None
    data = None
    for f in files:
        if 'json' in f:
            with open('./models/'+model_name+'/'+f) as json_file:
                data = json.load(json_file)
        if 'pth' in f:
            if USE_CPU == "TRUE":
                m = torch.load('./models/'+model_name+'/'+f, map_location=device)
            else:
                m = torch.load('./models/' + model_name + '/' + f)
            m.eval()
        model_dict[model_name] = {"weights": m, "labels": data}


@app.post('/predict')
async def run_model(model: str = Form(...), image: UploadFile = File(...)):
    im = Image.open(image.file).convert('RGB')
    image_tensor = test_transforms(im).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    m = model_dict[model]['weights']
    out = m(input)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    top_k_percentages = torch.topk(percentage, 5)
    response = []
    for i in range(0, 5):
        acc = top_k_percentages[0][i].item()
        label_index = top_k_percentages[1][i].item()
        predicted_class = model_dict[model]['labels'][label_index]
        classification = dict(
            [('Confidence', float(acc)), ('ObjectClass', predicted_class)])
        response.append(classification)
    return response