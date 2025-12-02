import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os

# -------------------------------
# 1. Modelo CNN simples
# -------------------------------
class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 16 * 16, 128)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x)

# -------------------------------
# 2. Função para processar imagem
# -------------------------------
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de imagem não encontrado: {path}")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
         raise Exception(f"Não foi possível carregar a imagem em: {path}.")

    img = cv2.resize(img, (64, 64))
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    return img

# -------------------------------
# 3. Comparador de Similaridade
# -------------------------------
def compare_images(img1_path, img2_path, model):
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    with torch.no_grad():
        emb1 = model(img1)
        emb2 = model(img2)

    dist = torch.norm(emb1 - emb2).item()

    return dist

# -------------------------------
# 4. Criação de Imagens Dummy
# -------------------------------
def create_dummy_images():
    def create_single_dummy_image(filename, shape='circle'):
        if os.path.exists(filename): return
        size = 64
        img = np.zeros((size, size, 1), dtype=np.uint8)
        center = (size // 2, size // 2)
        color = 255

        if shape == 'circle': cv2.circle(img, center, 20, color, -1)
        elif shape == 'square': cv2.rectangle(img, (center[0]-20, center[1]-20), (center[0]+20, center[1]+20), color, -1)
        elif shape == 'small_circle': cv2.circle(img, center, 15, color, -1)
        
        cv2.imwrite(filename, img)

    create_single_dummy_image("img1.png", shape='circle')
    create_single_dummy_image("img2.png", shape='square')
    create_single_dummy_image("img_semelhante.png", shape='small_circle')
    print("Arquivos de imagem de teste gerados: img1.png, img2.png, img_semelhante.png")
# -------------------------------
# 5. Exemplo de uso
# -------------------------------
if __name__ == '__main__':
    create_dummy_images()

    model = EmbeddingNet()
    
    dist_diferente = compare_images("img1.png", "img2.png", model)
    print("\n--- Comparação 1 (Diferentes) ---")
    print(f"Distância (Círculo vs Quadrado): {dist_diferente:.4f}")

    dist_semelhante = compare_images("img1.png", "img_semelhante.png", model)
    print("\n--- Comparação 2 (Semelhantes) ---")
    print(f"Distância (Círculo vs Círculo Menor): {dist_semelhante:.4f}")

    print("\n--- Conclusão ---")
    if dist_semelhante < dist_diferente:
        print("A distância entre imagens semelhantes foi menor.")
    else:
        print("A distância entre imagens semelhantes foi maior.")