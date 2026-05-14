# 🖼️ Deep Photo Style Transfer

**PyTorch · VGG19 · Neural Style Transfer · Image Processing**

Bu proje, bir **content image** görüntüsünün yapısını koruyarak başka bir **style image** görüntüsünün renk, ışık, kontrast ve atmosfer özelliklerini aktaran bir **Deep Photo Style Transfer** uygulamasıdır.

Amaç, klasik stil aktarım yöntemlerinde görülebilen aşırı yapay veya resim benzeri bozulmaları azaltarak daha doğal ve fotoğraf benzeri çıktılar elde etmektir.

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [İş Akışı](#-iş-akışı)
- [Klasör Yapısı](#-klasör-yapısı)
- [Kurulum](#️-kurulum)
- [Kullanım](#-kullanım)
- [Deney Parametreleri](#️-deney-parametreleri)
- [Çıktılar](#-çıktılar)
- [Kullanılan Teknolojiler](#-kullanılan-teknolojiler)
- [Notlar](#-notlar)

---

## 📌 Proje Hakkında

Bu projede iki temel görüntü kullanılır:

| Görsel | Açıklama |
|---|---|
| **Content Image** | Yapısı ve nesne düzeni korunacak ana görüntüdür. |
| **Style Image** | Renk, ışık, kontrast ve atmosfer özellikleri alınacak referans görüntüdür. |
| **Output Image** | Content yapısını koruyup style görüntüsünün atmosferini taşıyan sonuç görüntüsüdür. |

Projede önceden eğitilmiş **VGG19** modeli özellik çıkarıcı olarak kullanılır. Model yeniden eğitilmez; sadece content ve style görsellerinden özellik haritaları çıkarır.

Stil aktarımı sırasında çıktı görüntüsü, farklı loss fonksiyonları yardımıyla optimize edilir:

- **Content Loss:** Görüntünün yapısını korur.
- **Style Loss:** Renk, doku ve atmosfer aktarımını sağlar.
- **Total Variation Loss:** Gürültüyü azaltarak daha düzgün sonuçlar üretir.
- **Alpha Blending:** Çıktıyı content görüntüsüyle karıştırarak daha doğal görünüm sağlar.

Bu proje, **Deep Photo Style Transfer** makalesinden esinlenmiş sadeleştirilmiş bir öğrenci projesidir.

---

## 🏗 İş Akışı

```text
Content Image + Style Image
            │
            ▼
    Image Preprocessing
    Resize + Normalize
            │
            ▼
    Pretrained VGG19
    Feature Extraction
            │
            ▼
    Content Features
    Style Features
            │
            ▼
    Loss Calculation
    Content Loss + Style Loss + TV Loss
            │
            ▼
    Target Image Optimization
            │
            ▼
    Optional Alpha Blending
            │
            ▼
    Final Stylized Output
```

---

## 📁 Klasör Yapısı

```text
Deep-Photo-Style-Transfer/
│
├── images/
│   ├── content.jpg
│   └── style.jpg
│
├── outputs/
│   ├── result_exp1.png
│   ├── result_exp2.png
│   ├── result_exp3.png
│   └── comparison.png
│
├── main.py
├── styletransfer.py
├── utils.py
├── requirements.txt
└── README.md
```

### Dosya Açıklamaları

| Dosya / Klasör | Açıklama |
|---|---|
| `images/` | Content ve style görsellerinin bulunduğu klasör. |
| `outputs/` | Oluşturulan sonuç görsellerinin kaydedildiği klasör. |
| `main.py` | Projenin ana çalıştırma dosyası. |
| `styletransfer.py` | Stil aktarımı, loss hesaplamaları ve optimizasyon işlemlerini içerir. |
| `utils.py` | Görsel yükleme, kaydetme ve yardımcı işlemleri içerir. |
| `requirements.txt` | Gerekli Python kütüphanelerini içerir. |

---

## ⚙️ Kurulum

Projeyi çalıştırmadan önce gerekli kütüphaneleri yükleyin.

```bash
pip install -r requirements.txt
```

Eğer `requirements.txt` dosyası yoksa aşağıdaki temel kütüphaneleri kurabilirsiniz:

```bash
pip install torch torchvision pillow matplotlib numpy
```

---

## 🚀 Kullanım

Öncelikle `images/` klasörüne iki görsel ekleyin:

```text
images/content.jpg
images/style.jpg
```

Daha sonra terminal üzerinden projeyi çalıştırın:

```bash
python main.py
```

Program çalıştıktan sonra sonuçlar otomatik olarak `outputs/` klasörüne kaydedilir.

---

## ⚙️ Deney Parametreleri

Deney ayarları `main.py` dosyası içindeki `base_config` bölümünden değiştirilebilir.

| Parametre | Açıklama |
|---|---|
| `image_size` | İşlenecek görsel boyutunu belirler. |
| `num_steps` | Optimizasyon adım sayısını belirler. |
| `learning_rate` | Optimizasyon hızını belirler. |
| `content_weight` | Content görüntüsünün yapısının ne kadar korunacağını belirler. |
| `style_weight` | Style görüntüsünün etkisinin ne kadar güçlü olacağını belirler. |
| `tv_weight` | Gürültü azaltma etkisini belirler. |
| `alpha` | Alpha blending oranını belirler. |

---

## 📊 Çıktılar

Program çalıştırıldığında her deney için ayrı sonuç görüntüleri oluşturulur.

Örnek çıktı dosyaları:

```text
outputs/result_exp1.png
outputs/result_exp2.png
outputs/result_exp3.png
outputs/comparison.png
```

`comparison.png` dosyası, farklı deney sonuçlarını yan yana göstererek parametrelerin çıktı üzerindeki etkisini incelemeyi kolaylaştırır.

---

## 🔗 Kullanılan Teknolojiler

| Teknoloji | Amaç |
|---|---|
| Python | Ana programlama dili |
| PyTorch | Derin öğrenme altyapısı |
| Torchvision | Pretrained VGG19 modelini kullanmak için |
| VGG19 | Görüntülerden özellik çıkarmak için |
| Pillow | Görsel yükleme ve işleme |
| Matplotlib | Görsel kaydetme ve karşılaştırma |
| NumPy | Sayısal işlemler |

---

## 📝 Notlar

- VGG19 modeli yeniden eğitilmez, sadece özellik çıkarıcı olarak kullanılır.
- Optimize edilen şey model değil, çıktı görüntüsünün pikselleridir.
- Yüksek `style_weight` değeri stil etkisini artırır ancak görüntüde bozulmalara neden olabilir.
- Yüksek `content_weight` değeri content yapısını daha fazla korur.
- `alpha` değeri artırıldığında çıktı görüntüsü orijinal content görüntüsüne daha yakın olur.
- Daha yüksek `image_size` ve `num_steps` daha kaliteli sonuç verebilir fakat işlem süresini artırır.

---

## ✅ Sonuç

Bu proje, VGG19 tabanlı Neural Style Transfer yaklaşımı kullanarak bir content görüntüsüne başka bir style görüntüsünün atmosferini aktaran basitleştirilmiş bir Deep Photo Style Transfer uygulamasıdır.

Proje; derin öğrenme, görüntü işleme ve optimizasyon mantığını bir araya getirerek teknik olarak anlaşılır ve çalıştırılabilir bir stil aktarım sistemi sunar.
