from PIL import Image   
from torchvision import models

resnet_weights = models.ResNet50_Weights.DEFAULT
cats = resnet_weights.meta['categories']
transforms = resnet_weights.transforms()

model = models.resnet50(weights=resnet_weights )
# print(model)

img = Image.open('./image.png').convert('RGB')
img = transforms(img).unsqueeze(0)

model.eval()
p = model(img).squeeze()

res = p.softmax(dim=0).sort(descending=True)

for s, i in zip(res[0][:5], res[1][:5]):
    print(f"{cats[i]}: {s:.4f}")

# елена  - 8 951 503 21 28 (макс) - написать ФИО, яндекс маркет дата выхрода 12. собой одежду, в кармане ничего не должно быть. скзаать что от компании  UnitWork.  