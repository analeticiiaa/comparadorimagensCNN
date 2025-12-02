from comparador_imagens import EmbeddingNet, load_image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import cv2
import numpy as np

class TripletLoss(nn.Module):
    """
    Calcula a Triplet Loss: Max(0, d(A, P) - d(A, N) + margin)
    """
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)

        losses = F.relu(distance_positive - distance_negative + self.margin)
        
        return losses.mean()

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

    create_single_dummy_image("img_A.png", shape='circle') 
    create_single_dummy_image("img_P.png", shape='small_circle') 
    create_single_dummy_image("img_N.png", shape='square') 
    print("Arquivos de imagem de treinamento gerados: img_A.png, img_P.png, img_N.png")

if __name__ == '__main__':
    create_dummy_images()

    img_A = load_image("img_A.png")
    img_P = load_image("img_P.png")
    img_N = load_image("img_N.png")

    model = EmbeddingNet()
    criterion = TripletLoss(margin=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 50
    print("\n--- INÍCIO DO TREINAMENTO ---")

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        emb_A = model(img_A)
        emb_P = model(img_P)
        emb_N = model(img_N)

        loss = criterion(emb_A, emb_P, emb_N)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Época {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}")

    print("--- FIM DO TREINAMENTO ---")

    model.eval()

    with torch.no_grad():
        dist_AP = F.pairwise_distance(model(load_image("img_A.png")), model(load_image("img_P.png")), p=2).item()
        dist_AN = F.pairwise_distance(model(load_image("img_A.png")), model(load_image("img_N.png")), p=2).item()

    print("\n--- RESULTADOS PÓS-TREINAMENTO ---")
    print(f"Distância AP (Âncora vs Positivo - Semelhantes): {dist_AP:.4f}")
    print(f"Distância AN (Âncora vs Negativo - Diferentes): {dist_AN:.4f}")

    if dist_AP + 0.05 < dist_AN:
        print("SUCESSO! A distância entre semelhantes é visivelmente menor do que entre diferentes.")
    else:
        print("ATENÇÃO! A distância entre semelhantes ainda é maior ou igual. Tente aumentar as ÉPOCAS.")