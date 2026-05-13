import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

def load_image(image_path, max_size=400, shape=None):
    """
    Görüntüyü yükler ve model için uygun boyuta getirip PyTorch tensor'una dönüştürür.
    Görseller her zaman RGB formatında işlenir.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Hata: Görüntü dosyası bulunamadı: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    
    # Eğer shape verildiyse (style image için content image boyutu) ona göre boyutlandır
    if shape is not None:
        size = shape
    else:
        # Maksimum kenar uzunluğuna göre yeniden boyutlandır
        size = max_size if max(image.size) > max_size else max(image.size)
        
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # VGG19 için ImageNet istatistikleriyle normalizasyon
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])
    
    # 4 boyutlu tensor (Batch_Size=1, Channels=3, Height, Width)
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image

def im_convert(tensor):
    """
    Tensor formatındaki görüntüyü görselleştirmek veya kaydetmek için NumPy formatına dönüştürür.
    """
    # Görüntüyü CPU'ya alıp gradient takibinden çıkar
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    
    # (C, H, W) formatından (H, W, C) formatına çevir
    image = image.transpose(1, 2, 0)
    
    # Normalizasyonu geri al (Denormalize)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    
    # Piksellerin 0-1 aralığında olduğundan emin ol
    image = image.clip(0, 1)
    return image

def save_image(tensor, path):
    """
    Tensor görüntüsünü diske kaydeder.
    """
    # Dizin yoksa oluştur
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    image_np = im_convert(tensor)
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
    image_pil.save(path)
    
def blend_images(output_tensor, content_tensor, alpha=0.8):
    """
    Fotoğraf gerçekçiliğini artırmak için basit bir post-processing adımıdır.
    Oluşturulan stilize görüntüyü orijinal içerik görüntüsü ile belirli bir oranda (alpha) harmanlar.
    Bu, makaledeki (Deep Photo Style Transfer) orijinal photorealism regularization yaklaşımının 
    çok daha basit bir alternatifidir.
    
    alpha: Çıktı görüntünün ağırlığı (0.0 ile 1.0 arası). 
           (1.0 = Sadece çıktı, 0.0 = Sadece orijinal içerik)
    """
    blended = alpha * output_tensor + (1.0 - alpha) * content_tensor
    return blended

def create_comparison_plot(content_path, style_path, exp_results, save_path):
    """
    İçerik, stil ve tüm deney sonuçlarını yan yana gösteren bir görsel oluşturur.
    exp_results: List of tuples (label, image_path)
    """
    num_cols = 2 + len(exp_results) # 1 content, 1 style, N outputs
    
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))
    
    # Orijinal görseller
    axes[0].imshow(Image.open(content_path).convert('RGB'))
    axes[0].set_title("Content Image")
    axes[0].axis('off')
    
    axes[1].imshow(Image.open(style_path).convert('RGB'))
    axes[1].set_title("Style Image")
    axes[1].axis('off')
    
    # Deney sonuçları
    for i, (label, img_path) in enumerate(exp_results):
        axes[2+i].imshow(Image.open(img_path).convert('RGB'))
        axes[2+i].set_title(label)
        axes[2+i].axis('off')
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Bilgi] Karşılaştırma görseli kaydedildi: {save_path}")
