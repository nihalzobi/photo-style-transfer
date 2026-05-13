# Deep Photo Style Transfer

## Proje Amacı
Bu proje, bir içerik (content) görüntüsünün yapısal özelliklerini koruyarak, başka bir referans stil (style) görüntüsünün renk, ışık ve atmosfer özelliklerini içerik görüntüsüne aktaran bir **Neural Style Transfer** sistemidir. Klasik stil aktarım yöntemleri görüntülerde resim benzeri bozulmalar yaratırken, bu sistem fotoğraf gerçekçiliğini (photorealism) korumayı hedefler.

## Makale Bağlantısı
> **"Deep Photo Style Transfer", Luan et al., CVPR 2017**  
> Bu proje, ilgili makaleden esinlenilerek eğitim amaçlı geliştirilmiş basitleştirilmiş bir yaklaşımdır.

## Kullanılan Yöntem
Bu projede PyTorch kullanılarak önceden eğitilmiş (pretrained) VGG19 modeli bir özellik çıkarıcı (feature extractor) olarak kullanılmıştır. Projenin çalışma mantığı temel olarak bir hedef görüntünün kayıp fonksiyonları (Loss) aracılığıyla optimize edilmesine dayanır.

### Content Image ve Style Image Açıklaması
- **Content Image:** Yapısını, nesnelerini ve genel düzenini korumak istediğimiz temel fotoğraf.
- **Style Image:** Renk paletini, kontrastını ve atmosferini (stilini) almak istediğimiz referans fotoğraf.

### VGG19'un Projedeki Rolü
VGG19 modeli, görüntülerin yüksek seviyeli ve düşük seviyeli özelliklerini (kenarlar, dokular, şekiller) çıkarmak için kullanılmıştır. Model eğitilmemiş, ağırlıkları dondurulmuştur (`requires_grad = False`).

### Loss Fonksiyonları Nelerdir?
- **Content Loss:** Hedef görüntünün, içerik görüntüsü ile aynı fiziksel yapıya ve şekillere sahip olmasını zorlar. (VGG19 `conv4_2` katmanı kullanılır).
- **Style Loss:** Hedef görüntünün, stil görüntüsünün renk ve doku özelliklerine sahip olmasını sağlar. (VGG19 `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1` katmanları kullanılır).
- **Gram Matrix:** Stil kaybını hesaplamak için kullanılır. Bir katmandaki farklı özellik haritalarının birbirleriyle olan matematiksel korelasyonunu (ilişkisini) ifade eder ve görüntünün "stil" bilgisini temsil eder.
- **Total Variation Loss (TV Loss):** Neden eklendi? Optimize edilen hedef görüntüde oluşabilecek yüksek frekanslı gürültüyü (noise) engellemek, komşu pikseller arasındaki uyumu artırarak daha pürüzsüz ve fotoğraf benzeri (photorealistic) sonuçlar almak için eklenmiştir.

### Makaledeki Yöntemden Farkı Nedir?
Bu proje, Deep Photo Style Transfer makalesinden esinlenmiştir. Makaledeki orijinal yöntem photorealism regularization için Matting Laplacian ve semantic segmentation kullanmaktadır. Bu projede ise daha uygulanabilir bir öğrenci projesi kapsamında VGG19 tabanlı Neural Style Transfer uygulanmış, fotoğraf gerçekçiliğini artırmak için **total variation loss** ve opsiyonel **content-output blending (Alpha Blending)** kullanılmıştır.

## Kurulum ve Çalıştırma

### Gereksinimler
Gereksinimleri yüklemek için terminalde proje dizinindeyken şu komutu çalıştırın:
```bash
pip install -r requirements.txt
```

### Nasıl Çalıştırılır?
1. `images/` klasörünün içerisine `content.jpg` ve `style.jpg` adında iki adet görsel yerleştirin.
2. Ana dosyayı çalıştırın:
```bash
python main.py
```

### Çıktılar Nerede Oluşur?
Tüm çıktı dosyaları ve deney sonuçları `outputs/` klasöründe oluşacaktır. İşlem bitiminde `outputs/comparison.png` dosyasını açarak 3 farklı deneyin sonucunu yan yana görebilirsiniz.

## Deney Parametreleri Nasıl Değiştirilir?
`main.py` dosyası içindeki `base_config` sözlüğünden (dictionary) aşağıdaki parametreleri değiştirebilirsiniz:
- `image_size`: Çözünürlüğü artırmak veya azaltmak için.
- `num_steps`: Eğitim adım sayısı.
- `learning_rate`: Optimizasyon hızı.
- `content_weight` ve `tv_weight`: Ağırlık dengeleri.
- `alpha`: Alpha blending için orijinal görüntünün baskınlık oranı.

Ayrıca `experiments` listesi içerisinden farklı `style_weight` değerleri ekleyerek yeni deneyler tanımlayabilirsiniz.

## Projenin Sınırlamaları ve Gelecek Geliştirmeler
- **Sınırlamalar:** Matting Laplacian ve Semantic Segmentation kullanılmadığı için, stil aktarımı gökyüzü veya bina gibi spesifik nesneleri ayırt etmeden tüm görüntüye uygulanır. Renk taşmaları yaşanabilir.
- **Gelecekte Yapılabilecekler:**
  - Semantic segmentation eklenebilir.
  - Makaledeki asıl Matting Laplacian uygulanabilir.
  - Daha hızlı çalışması için Feed-Forward Style Transfer (Johnson et al.) modeli denenebilir.
  - Farklı content/style görüntü çiftleriyle test yapılabilir.
