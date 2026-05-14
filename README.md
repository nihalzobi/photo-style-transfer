# 🖼️ Deep Photo Style Transfer

VGG19 · Neural Style Transfer · PyTorch · Gram Matrix · Total Variation Loss · Alpha Blending kullanılarak geliştirilen fotoğraf tabanlı stil aktarım sistemi.

Bu proje, bir içerik görüntüsünün yapısını koruyarak başka bir referans stil görüntüsünün renk, ışık, kontrast ve atmosfer özelliklerini aktarmayı amaçlar. Klasik Neural Style Transfer yöntemleri çoğu zaman resim benzeri veya yapay görünümlü çıktılar üretirken, bu projede daha doğal ve fotoğraf benzeri sonuçlar elde etmek hedeflenmiştir.

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Referans Makale](#-referans-makale)
- [Sistem Mimarisi](#-sistem-mimarisi)
- [Kullanılan Yöntem](#-kullanılan-yöntem)
- [Loss Fonksiyonları](#-loss-fonksiyonları)
- [Klasör Yapısı](#-klasör-yapısı)
- [Kurulum](#️-kurulum)
- [Kullanım](#-kullanım)
- [Deney Parametreleri](#-deney-parametreleri)
- [Çıktılar](#-çıktılar)
- [Dosya Açıklamaları](#-dosya-açıklamaları)
- [Notlar ve Sınırlamalar](#-notlar-ve-sınırlamalar)
- [Gelecek Geliştirmeler](#-gelecek-geliştirmeler)

---

## 📌 Proje Hakkında

Bu sistem, kullanıcının verdiği iki görsel üzerinden otomatik olarak stil aktarımı yapar:

- **Content Image:** Yapısını, nesnelerini ve genel kompozisyonunu korumak istediğimiz ana görüntü.
- **Style Image:** Renk paleti, ışık, kontrast ve atmosfer özelliklerini almak istediğimiz referans görüntü.
- **Output Image:** Content görüntüsünün yapısını koruyan, style görüntüsünün görsel atmosferini taşıyan yeni görüntü.

Örneğin gündüz çekilmiş bir doğa fotoğrafına, gün batımı temalı başka bir görselin sıcak renkleri ve ışık atmosferi aktarılabilir.

Bu proje özellikle **Deep Learning**, **Computer Vision** ve **Image Processing** konularını birleştiren bir uygulamadır.

---

## 📄 Referans Makale

Bu proje aşağıdaki çalışmadan esinlenilerek geliştirilmiştir:

**Deep Photo Style Transfer**  
Luan et al., CVPR 2017

Orijinal makalede fotoğraf gerçekçiliğini korumak için:

- Matting Laplacian
- Semantic Segmentation
- Photorealism Regularization

gibi gelişmiş yöntemler kullanılmaktadır.

Bu projede ise öğrenci projesi kapsamında daha sade, anlaşılır ve uygulanabilir bir yaklaşım tercih edilmiştir. Bu nedenle proje, makalenin birebir kopyası değildir. Makaleden ilham alan, **VGG19 tabanlı basitleştirilmiş bir Neural Style Transfer uygulamasıdır.**

---

## 🏗 Sistem Mimarisi

```text
Content Image + Style Image
            │
            ▼
    ┌────────────────────┐
    │ Image Preprocessing │
    │ Resize + Normalize  │
    └────────────────────┘
            │
            ▼
    ┌────────────────────┐
    │ Pretrained VGG19    │
    │ Feature Extractor   │
    └────────────────────┘
            │
            ▼
    ┌────────────────────────────┐
    │ Feature Maps Extraction     │
    │ Content + Style Layers      │
    └────────────────────────────┘
            │
            ▼
    ┌────────────────────────────┐
    │ Loss Calculation            │
    │ Content Loss                │
    │ Style Loss                  │
    │ Total Variation Loss        │
    └────────────────────────────┘
            │
            ▼
    ┌────────────────────────────┐
    │ Target Image Optimization   │
    │ Pixel-based Optimization    │
    └────────────────────────────┘
            │
            ▼
    ┌────────────────────────────┐
    │ Alpha Blending              │
    │ Optional Naturalization     │
    └────────────────────────────┘
            │
            ▼
      Final Stylized Image
