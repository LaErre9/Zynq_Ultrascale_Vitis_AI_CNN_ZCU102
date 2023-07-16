import random
import os, cv2, PIL
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.model = nn.Sequential(
            # 1st convolutional network Layers
            nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),
            
            # 2nd convolutional network Layers
            nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),
            
            # 3rd convolutional network Layers
            nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),
            
            # 4th convolutional network Layers
            nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),
            
            # 5th convolutional network Layers
            nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),
            
            nn.Flatten(),
            
            # feed forward Layers
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, 43)
            )
    def forward(self, x):
        x = self.model(x)
        return x


def test(model, device, folder_path):
    '''
    test the model
    '''
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        # Ottenere la lista di tutte le immagini nella cartella
        all_images = []
        for filename in os.listdir(folder_path):
            dir_path = os.path.join(folder_path, filename)
            for image in os.listdir(dir_path):
                image_path = os.path.join(dir_path, image)
                all_images.append((image_path, int(filename)))  # Salva il percorso dell'immagine e l'etichetta associata

        # Mescola le immagini in modo casuale
        random.shuffle(all_images)

        # Selezionare casualmente un campione di immagini
        random_images = random.sample(all_images, 10000)  # Cambia 1000 con il numero desiderato di immagini da valutare
        for image_path, true_label in random_images:
            # Make prediction for the current image
            if type(image_path) == str:
                img = torchvision.transforms.ToTensor()(PIL.Image.open(image_path))
            img = cv2.resize(img.permute(1,2,0).numpy(),(32,32))
            img = torch.from_numpy(img).permute(2,0,1)
            img_tensor = img.unsqueeze(0).to(device)
            pred_label = int(model(img_tensor).argmax(axis=1)[0])

            # Check if the prediction is correct
            if pred_label == true_label:
                correct += 1

            total += 1
            print('\rNumero di immagini valutate: {}'.format(total), end='')

    acc = 100. * correct / total
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, total, acc))

    return

''' image transformation for image generation '''
gen_transform = torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor()
                           ])
