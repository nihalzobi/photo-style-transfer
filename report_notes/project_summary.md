# Proje Özeti: Basitleştirilmiş Deep Photo Style Transfer

## Problem Tanımı
Geleneksel Neural Style Transfer (Sinirsel Stil Aktarımı) yöntemleri, bir görüntünün stilini diğerine aktarırken başarılı olsa da, genellikle sonuçlarda resim veya tablo benzeri (painterly) bozulmalar, fırça darbeleri ve geometrik deformasyonlar oluşturur. Eğer amacımız bir "fotoğrafın" atmosferini başka bir "fotoğrafa" aktarmaksa (örneğin gündüz çekilmiş bir fotoğrafı gece atmosferine sokmak), bu resimsi bozulmalar istenmeyen bir durumdur. Problem, fotoğrafın fiziksel gerçekliğini bozmadan renk ve ışık özelliklerinin değiştirilmesidir.

## Kullanılan Makale
**Makale Adı:** Deep Photo Style Transfer  
**Yazarlar:** Fujun Luan, Sylvain Paris, Eli Shechtman, Kavita Bala  
**Yayın:** CVPR 2017

## Yöntemin Genel Mantığı
Yöntem, derin öğrenme tabanlı bir optimizasyon sürecidir. İstenen sonuca ulaşmak için model eğitilmez; bunun yerine rastgele veya önceden belirlenmiş bir "hedef görüntü" piksel piksel güncellenir. Bu güncelleme süreci, hedeflenen yapıyı korumak ve istenen stili elde etmek için hesaplanan "Kayıp (Loss)" değerlerinin minimize edilmesine dayanır.

## Kullanılan Model: VGG19
Görüntülerin özelliklerini (feature) anlamak ve çıkarmak için önceden eğitilmiş VGG19 Convolutional Neural Network modeli kullanılmıştır. Modelin sınıflandırma (classifier) kısmı atılmış, sadece özellik çıkarma (feature extraction) katmanları alınmıştır. Ağırlıklar dondurulmuş olup, model eğitim sürecine girmez, sadece bir referans ölçüm aracı olarak görev yapar.

## Content Loss Açıklaması
Content Loss (İçerik Kaybı), üretilen çıktının orijinal içerik (content) fotoğrafına yapısal olarak benzemesini sağlar. VGG19'un daha derin bir katmanından (örneğin `conv4_2`) alınan özellik haritalarının Ortalama Kare Hatası (MSE) hesaplanarak bulunur. Bu sayede nesnelerin yerleri ve genel sahne korunur.

## Style Loss Açıklaması
Style Loss (Stil Kaybı), üretilen çıktının renk, ışık ve doku açısından stil (style) fotoğrafına benzemesini sağlar. VGG19'un birden fazla katmanından (`conv1_1` ile `conv5_1` arası) özellikler çekilerek Gram Matrisleri üzerinden hesaplanır.

## Gram Matrix Açıklaması
Gram Matrix, bir katmandaki farklı özellik haritalarının birbirleriyle olan iç çarpımını hesaplayan bir matristir. Bu çarpım, uzamsal (spatial) yapıyı yok edip geriye sadece "hangi özelliklerin birlikte görüldüğü" (örneğin kırmızı renk ile düz dokunun ilişkisi) bilgisini bırakır. Bu da matematiksel olarak görüntünün "stil" bilgisini temsil eder.

## Total Variation Loss Açıklaması
Görüntüdeki yüksek frekanslı gürültüyü azaltan bir düzenleme (regularization) tekniğidir. Komşu pikseller arasındaki renk değişimlerini cezalandırarak, daha pürüzsüz ve homojen bir görüntü oluşturulmasını sağlar. Fotoğraf gerçekçiliğine (photorealism) katkıda bulunur.

## Deneyler
Sistemin farklı stil yoğunluklarına nasıl tepki verdiğini gözlemlemek için, `style_weight` parametresi değiştirilerek 3 farklı deney tasarlanmıştır:
1. Düşük Stil Ağırlığı (Sadece hafif renk değişimi)
2. Orta Stil Ağırlığı (Dengeli yapı ve stil)
3. Yüksek Stil Ağırlığı (Stil görüntüsünün baskın renklerinin güçlü bir şekilde aktarımı)

## Sonuçların Değerlendirilmesi
Elde edilen çıktılar, total variation loss ve alpha blending teknikleri sayesinde klasik Neural Style Transfer yöntemine göre daha az deformasyona sahip, daha gerçekçi sonuçlar vermiştir. Farklı `style_weight` değerleri, kullanıcının ihtiyacına göre stilin yoğunluğunu ayarlayabilmesine olanak tanır.

## Makaledeki Deep Photo Style Transfer ile Benzerlikler
- Özellik çıkarımı için aynı şekilde VGG19 mimarisi kullanılmıştır.
- Content loss için benzer katman (`conv4_2`) tercih edilmiştir.
- Style loss hesaplamasında çoklu katmanlar ve Gram Matrisi mantığı aynıdır.
- Temel amaç fotoğraf gerçekçiliğini (photorealism) sağlamaktır.

## Makaledeki Yöntemden Farklılıklar
- **Matting Laplacian Eksikliği:** Orijinal makaledeki karmaşık ve ağır hesaplamalı Matting Laplacian düzenlemesi yerine, uygulanması çok daha pratik olan Total Variation Loss tercih edilmiştir.
- **Semantic Segmentation Eksikliği:** Orijinal makale, gökyüzünün stili gökyüzüne, suyun stili suya gitsin diye maskeleme yapar. Bu projede maskeleme (segmentation) yoktur, stil tüm görüntüye global olarak uygulanır.
- **Optimizasyon:** L-BFGS yerine daha yaygın ve parametre ayarı kolay olan Adam optimizasyon algoritması kullanılmıştır.
- **Post-processing:** Fotoğraf gerçekçiliğini kurtarmak için Alpha Blending adı verilen basit bir harmanlama eklenmiştir.

## Sınırlamalar
- Semantic segmentation olmadığı için renk sızıntıları (örneğin gökyüzündeki maviliğin binalara geçmesi) yaşanabilir.
- Model her adımda pikselleri güncellediği için yüksek çözünürlüklerde işlem süresi uzun olabilir ve yüksek RAM/VRAM gerektirebilir.

## Sonuç
Bu proje, karmaşık "Deep Photo Style Transfer" yönteminin temel fikirlerini alarak anlaşılır, eğitici ve uygulanabilir bir Python projesi haline getirmiştir. İleri düzey matematiksel işlemler (Matting Laplacian) atlanmasına rağmen, yapılan optimizasyonlar ve kullanılan kayıp fonksiyonları ile geleneksel NST'ye kıyasla daha "fotoğraf benzeri" başarılı sonuçlar elde edilmektedir.
