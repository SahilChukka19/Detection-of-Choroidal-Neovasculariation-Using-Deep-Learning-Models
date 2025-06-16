from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import torch
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import torchvision.models as models
import torch.nn as nn

app = Flask(__name__)

dic = {0 : 'cnv', 1 : 'normal'}

#cnn model
cnn_model = load_model('cnn/model-cnn70-30.h5')
cnn_model.make_predict_function()

#resnet18
resnet18_model = models.resnet18(pretrained=False)  # You may set pretrained=True if your model was originally pretrained
resnet18_model.fc = torch.nn.Linear(in_features=512, out_features=2)  # Adjust the number of output features
# Load the model's state_dict
resnet18_model_path = 'resnet18/model-resnet18-70-30.pth'
resnet18_model.load_state_dict(torch.load(resnet18_model_path, map_location=torch.device('cpu')))
resnet18_model.eval()
# Define the image preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

#resnet50
resnet50_model = load_model('resnet50/model-resnet50-80-20.h5')
resnet50_model.make_predict_function()

#vggnet16
vggnet16_model = load_model('vggnet16/model-vggnet16-80-20.h5')
vggnet16_model.make_predict_function()

#vggnet19
vggnet19_model = load_model('vggnet19/model-vggnet19-60-40.h5')
vggnet19_model.make_predict_function()

#InceptionV3
inceptionv3_model = load_model('inceptionv3/inceptionv3_model-60-40.h5')
inceptionv3_model.make_predict_function()

#mobilenetv2
mobilenetv2_model = load_model('mobilenetv2/mpmv80-20.h5')
mobilenetv2_model.make_predict_function()

#efficientnetv2l
efficientnetv2l_model = load_model('efficientnetv2l/efficientnetv2_model-70-30.h5')
efficientnetv2l_model.make_predict_function()

#vit
import torch
import torch.nn as nn
import torchvision.models as models


class ViTModel(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(ViTModel, self).__init__()
        # Load ViT base model (ViT-B/16)
        self.vit = models.vit_b_16(pretrained=False)

        # Replace the classification head
        self.vit.heads = nn.Linear(768, num_classes)

        # Initialize the classification head
        nn.init.normal_(self.vit.heads.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.vit.heads.bias)

        # Ensure head is trainable
        for param in self.vit.heads.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.vit(x)


def load_vit_model(model_path: str, num_classes: int = 2, device: str = 'cpu') -> nn.Module:
    """
    Loads a Vision Transformer (ViT) model from checkpoint.

    Args:
        model_path (str): Path to the saved .pth file
        num_classes (int): Number of output classes
        device (str): 'cpu' or 'cuda'

    Returns:
        nn.Module: Loaded ViT model ready for inference
    """
    model = ViTModel(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


# Usage
vit_model_path = 'vit/vit_model-60-40.pth'
vit_model = load_vit_model(model_path=vit_model_path, num_classes=2, device='cpu')


def predict_label(model, image_path):
    if model == 'cnn':
        i = image.load_img(image_path, target_size=(256, 256))
        i = image.img_to_array(i) / 255.0
        i = i.reshape(1, 256, 256, 3)
        p = cnn_model.predict(i)
        predicted_class = dic[int(p[0][0])]
    elif model == 'resnet18':
        input_image = preprocess_image(image_path)
        with torch.no_grad():
            output = resnet18_model(input_image)
        predicted_class = dic[torch.argmax(output).item()]
    elif model == 'vggnet16':
        i = image.load_img(image_path, target_size=(224, 224))
        i = image.img_to_array(i) / 255.0
        i = i.reshape(1, 224, 224, 3)
        p = vggnet16_model.predict(i)
        predicted_class = dic[int(p[0][0])]
    elif model == 'vggnet19':
        i = image.load_img(image_path, target_size=(224, 224))
        i = image.img_to_array(i) / 255.0
        i = i.reshape(1, 224, 224, 3)
        p = vggnet19_model.predict(i)
        predicted_class = dic[int(p[0][0])]
    elif model == 'inceptionv3':
        i = image.load_img(image_path, target_size=(150, 150))
        i = image.img_to_array(i) / 255.0
        i = i.reshape(1, 150, 150, 3)
        p = inceptionv3_model.predict(i)
        predicted_class = dic[int(p[0][0])]
    elif model == 'mobilenetv2':
        i = image.load_img(image_path, target_size=(224, 224))
        i = image.img_to_array(i) / 255.0
        i = i.reshape(1, 224, 224, 3)
        p = mobilenetv2_model.predict(i)
        predicted_class = dic[int(p[0][0])]
    elif model == 'efficientnetv2l':
        i = image.load_img(image_path, target_size=(224, 224))
        i = image.img_to_array(i) / 255.0
        i = i.reshape(224, 224,3)
        i = np.expand_dims(i, axis=0)
        p = efficientnetv2l_model.predict(i)
        predicted_class = dic[np.argmax(p[0])]
    elif model == 'resnet50':
        i = image.load_img(image_path, target_size=(224, 224))
        i = image.img_to_array(i) / 255.0
        i = i.reshape(1, 224, 224, 3)
        p = resnet50_model.predict(i)
        predicted_class = dic[int(p[0][0])]
    elif model == 'vit':  # Add ViT model case
        input_image = preprocess_image(image_path)
        with torch.no_grad():
            output = vit_model(input_image)
        predicted_class = dic[torch.argmax(output).item()]
    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        selected_model = request.form.get('model')
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        predicted_class = predict_label(selected_model, img_path)
       
        if selected_model == 'cnn':
            model_info = "CNN"
        elif selected_model == 'resnet18':
            model_info = "ResNet18"
        elif selected_model == 'vggnet16':
            model_info = "VggNet16"
        elif selected_model == 'vggnet19':
            model_info = "VggNet19"
        elif selected_model == 'inceptionv3':
            model_info = "InceptionV3"
        elif selected_model == 'mobilenetv2':
            model_info = "MobileNetV2"
        elif selected_model == 'efficientnetv2l':
            model_info = "efficientnetv2l"
        elif selected_model == 'vit':
            model_info = "Vision Transformer"
        elif selected_model == 'resnet50':
            model_info = "ResNet50"

    return render_template("prediction.html", prediction=predicted_class, model_info=model_info, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)