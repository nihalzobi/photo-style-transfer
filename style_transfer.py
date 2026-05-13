import torch
import torch.nn as nn
from torchvision import models

def get_vgg19():
    """
    Sadece Feature Extractor olarak kullanılacak önceden eğitilmiş VGG19 modelini yükler.
    Modelin sınıflandırma (classifier) katmanları atılır, sadece feature katmanları alınır.
    Model eğitilmeyeceği için parametreleri dondurulur.
    """
    # Güncel torchvision sürümlerinde VGG19_Weights kullanılıyor.
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
    
    # Modelin ağırlıklarını donduruyoruz (Geri yayılımda güncellenmeyecek)
    for param in vgg.parameters():
        param.requires_grad = False
        
    return vgg

def get_features(image, model, layers=None):
    """
    Görüntüyü VGG modelinden geçirerek belirli katmanlardan feature haritalarını çıkarır.
    """
    # VGG19'daki katman indeksleri ve kullanacağımız isimleri:
    if layers is None:
        layers = {
            '0': 'conv1_1',   # Style loss
            '5': 'conv2_1',   # Style loss
            '10': 'conv3_1',  # Style loss
            '19': 'conv4_1',  # Style loss
            '21': 'conv4_2',  # Content loss (İçerik yapısını korumak için)
            '28': 'conv5_1'   # Style loss
        }
        
    features = {}
    x = image
    # Modeldeki tüm katmanlardan geçerek sadece istediğimiz katmanların çıktılarını alıyoruz
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

def gram_matrix(tensor):
    """
    Style loss hesaplamasında kullanılacak olan Gram Matrix'i oluşturur.
    Gram Matrix, bir katmandaki farklı özellik haritalarının (feature maps)
    birbirleriyle olan korelasyonunu (ilişkisini) gösterir. Bu yapı, 
    görüntünün yerel stil (doku, renk vb.) bilgisini matematiksel olarak temsil eder.
    """
    # Batch size, depth (channel), height, width
    b, d, h, w = tensor.size()
    
    # Tensoru düzleştirip 2 boyutlu hale getiriyoruz: (channels, height * width)
    tensor = tensor.view(b * d, h * w)
    
    # Tensor ile kendi devriğini (transpose) çarpıyoruz -> Gram Matrisi
    gram = torch.mm(tensor, tensor.t())
    
    return gram

def content_loss_fn(target_features, content_features):
    """
    Content Loss: Hedef (çıktı) görüntünün içeriğinin orijinal içerik görüntüsüne
    yapısal olarak benzemesini sağlar.
    Burada Ortalama Kare Hata (Mean Squared Error - MSE) kullanıyoruz.
    """
    return torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

def style_loss_fn(target_features, style_grams, style_weights_dict):
    """
    Style Loss: Hedef (çıktı) görüntünün stilinin, stil görüntüsünün özelliklerine
    benzemesini sağlar. Bunu, her katmanın Gram matrisleri arasındaki farkı
    hesaplayarak (MSE) yapıyoruz.
    """
    style_loss = 0
    for layer in style_weights_dict:
        # Hedef görüntünün o katmandaki özelliklerini al ve Gram matrisini çıkar
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        
        # Orijinal stil Gram matrisi
        style_gram = style_grams[layer]
        
        # Katman boyutları (Kayıp değerini normalize etmek için)
        _, d, h, w = target_feature.shape
        
        # MSE Loss hesapla ve katmanın kendi ağırlığı ile çarp
        layer_style_loss = style_weights_dict[layer] * torch.mean((target_gram - style_gram)**2)
        style_loss += layer_style_loss / (d * h * w)
        
    return style_loss

def total_variation_loss(img):
    """
    Total Variation Loss (TV Loss): 
    Görüntüdeki yüksek frekanslı gürültüyü (noise) azaltmaya yarayan regularization kaybıdır.
    Bitişik pikseller arasındaki farkları minimize ederek daha pürüzsüz,
    fotoğraf benzeri sonuçlar elde etmemize yardımcı olur.
    
    [Akademik Not]: Bu yaklaşım, Deep Photo Style Transfer (Luan et al.) makalesindeki 
    karmaşık 'Matting Laplacian' tabanlı photorealism regularization'ın daha basit 
    ve kolay uygulanabilir bir alternatifidir. Birebir aynı şeyi yapmasa da fotoğraf
    gerçekçiliğini artırmak için oldukça etkilidir.
    """
    # Görüntünün komşu pikselleri arasındaki fark (Hem yatay hem dikey)
    tv_h = torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w
