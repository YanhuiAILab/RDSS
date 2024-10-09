import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import CLIPVisionModel, CLIPVisionConfig
import pickle
from tqdm import tqdm


# set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# initialize CLIP
config = CLIPVisionConfig.from_pretrained("./clip-vit-base-patch32")
model = CLIPVisionModel(config).to(device)
print("CLIP Loaded Successfully")

model.eval()

# load and preprocess CIFAR-10/100
# modify the code, so that it can handle CIFAR-10 and CIFAR-100
transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()])
cifar100_dataset = datasets.CIFAR100(
    root='./data/cifar100', train=True, download=False, transform=transform)
print("Load CIFAR-100 Successfully")

data_loader = DataLoader(dataset=cifar100_dataset,
                         batch_size=1024, shuffle=False, num_workers=2)

# define a dictionary to store image embeddings
image_embeddings = {}

# get image embeddings
for i, (images, labels) in tqdm(enumerate(data_loader)):
    images = images.to(device)

    with torch.no_grad():
        image_features = model(images).pooler_output

    for j, image_feature in enumerate(image_features):
        image_embeddings[i*data_loader.batch_size +
                         j] = image_feature.cpu().numpy()

# save image embeddings
with open('image_embeddings_cifar_100.pkl', 'wb') as f:
    pickle.dump(image_embeddings, f)

with open('image_embeddings_cifar_100.pkl', 'rb') as f:
    image_embeddings = pickle.load(f)
