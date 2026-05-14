# Deep Photo Style Transfer

Bu proje, bir fotoğrafın yapısını koruyarak başka bir fotoğrafın renk, ışık, kontrast ve atmosfer özelliklerini aktaran bir **Neural Style Transfer** uygulamasıdır.

Projenin temel amacı, klasik stil aktarım yöntemlerinde sık görülen aşırı bozulmaları azaltarak daha doğal ve fotoğraf benzeri sonuçlar üretmektir. Bu nedenle proje, yalnızca sanatsal bir efekt oluşturmayı değil, aynı zamanda **photorealistic style transfer** mantığını basitleştirilmiş şekilde göstermeyi hedefler.

---

## Project Overview

Deep Photo Style Transfer projesinde iki farklı görüntü kullanılır:

- **Content Image:** Yapısını, nesnelerini ve genel düzenini korumak istediğimiz ana görüntüdür.
- **Style Image:** Renk paleti, ışık, kontrast ve atmosfer özelliklerini almak istediğimiz referans görüntüdür.

Model, content görüntüsünün genel yapısını korurken style görüntüsünün görsel özelliklerini hedef görüntüye aktarmaya çalışır.

Örneğin gündüz çekilmiş bir manzara fotoğrafına, gün batımı temalı başka bir fotoğrafın sıcak renkleri ve atmosferi aktarılabilir.

---

## Reference Paper

Bu proje aşağıdaki makaleden esinlenilerek geliştirilmiştir:

**Deep Photo Style Transfer**  
Luan et al., CVPR 2017

Orijinal makalede fotoğraf gerçekçiliğini korumak için **Matting Laplacian** ve **semantic segmentation** gibi gelişmiş yöntemler kullanılmaktadır.

Bu projede ise öğrenci projesi kapsamında daha sade ve uygulanabilir bir yaklaşım tercih edilmiştir. Bu nedenle proje, makalenin birebir uygulaması değil; makaleden ilham alan, VGG19 tabanlı basitleştirilmiş bir Neural Style Transfer uygulamasıdır.

---

## Features

- Content ve style görüntüleriyle stil aktarımı yapar.
- Önceden eğitilmiş VGG19 modelini feature extractor olarak kullanır.
- Content Loss ile görüntünün yapısını korur.
- Style Loss ile renk, doku ve atmosfer aktarımı yapar.
- Gram Matrix kullanarak stil bilgisini temsil eder.
- Total Variation Loss ile gürültüyü azaltır.
- Alpha Blending ile daha doğal ve dengeli sonuçlar üretir.
- Farklı deney parametreleriyle birden fazla çıktı oluşturabilir.
- Sonuçları `outputs/` klasörüne kaydeder.
- Deney sonuçlarını karşılaştırmak için `comparison.png` çıktısı üretir.

---

## Technologies Used

Projede kullanılan temel teknolojiler:

- Python
- PyTorch
- Torchvision
- VGG19 Pretrained Model
- Pillow
- Matplotlib
- NumPy

---

## How It Works

Projenin çalışma mantığı şu şekildedir:

1. Content ve style görüntüleri yüklenir.
2. Görüntüler belirlenen boyuta getirilir.
3. Önceden eğitilmiş VGG19 modeli kullanılarak görüntülerden özellikler çıkarılır.
4. Hedef görüntü başlangıçta content görüntüsünün kopyası olarak alınır.
5. Hedef görüntü, loss fonksiyonlarına göre optimize edilir.
6. Optimizasyon sonunda content yapısını koruyan ve style atmosferini taşıyan yeni bir görüntü elde edilir.
7. Çıktılar `outputs/` klasörüne kaydedilir.

Bu projede VGG19 modeli yeniden eğitilmez. Modelin ağırlıkları dondurulur ve yalnızca özellik çıkarıcı olarak kullanılır. Optimize edilen şey model değil, doğrudan hedef görüntünün pikselleridir.

---

## Role of VGG19

Projede VGG19 modeli, görüntülerin farklı seviyelerdeki özelliklerini çıkarmak için kullanılır.

VGG19 sayesinde görüntülerden şu tür bilgiler elde edilir:

- Kenarlar
- Renk geçişleri
- Dokular
- Şekiller
- Nesne yapıları
- Yüksek seviyeli görsel özellikler

Modelin ağırlıkları sabit tutulur:

```python
requires_grad = False
