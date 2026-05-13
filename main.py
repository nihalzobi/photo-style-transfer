import torch
import torch.optim as optim
import os

from utils import load_image, save_image, blend_images, create_comparison_plot
from style_transfer import (
    get_vgg19,
    get_features,
    gram_matrix,
    content_loss_fn,
    style_loss_fn,
    total_variation_loss
)


def run_experiment(config):
    """
    Belirli parametrelerle (config) bir adet stil transferi deneyi çalıştırır.
    """

    print(f"\n--- Deney Başlıyor: {config['label']} ---")
    print(
        f"Parametreler: "
        f"Style Weight: {config['style_weight']}, "
        f"Content Weight: {config['content_weight']}, "
        f"TV Weight: {config['tv_weight']}"
    )

    # 1. Cihaz kontrolü
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    # 2. VGG19 modelini yükle
    vgg = get_vgg19().to(device).eval()

    # 3. Görüntüleri yükle
    content = load_image(config['content_path'], max_size=config['image_size']).to(device)

    # Style image, content image ile aynı boyutta yüklensin
    style_shape = [content.shape[2], content.shape[3]]
    style = load_image(config['style_path'], shape=style_shape).to(device)

    # 4. Content ve style feature değerlerini hesapla
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {
        layer: gram_matrix(style_features[layer])
        for layer in style_features
    }

    # 5. Output image başlatma
    # Fotoğraf gerçekçiliğini korumak için random noise yerine content image kopyası ile başlatıyoruz.
    target = content.clone().requires_grad_(True).to(device)

    # 6. Style katman ağırlıkları
    style_weights_dict = {
        'conv1_1': 1.5,
        'conv2_1': 1.2,
        'conv3_1': 1.0,
        'conv4_1': 0.8,
        'conv5_1': 0.5
    }

    # 7. Optimizer
    optimizer = optim.Adam([target], lr=config['learning_rate'])

    # 8. Eğitim / optimizasyon döngüsü
    for step in range(1, config['num_steps'] + 1):
        target_features = get_features(target, vgg)

        content_loss = content_loss_fn(target_features, content_features)
        style_loss = style_loss_fn(target_features, style_grams, style_weights_dict)
        tv_loss = total_variation_loss(target)

        total_loss = (
            (config['content_weight'] * content_loss)
            + (config['style_weight'] * style_loss)
            + (config['tv_weight'] * tv_loss)
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == config['num_steps']:
            print(
                f"Adım {step}/{config['num_steps']} | "
                f"Total Loss: {total_loss.item():.2f} | "
                f"Content: {content_loss.item():.2f} | "
                f"Style: {style_loss.item():.2f} | "
                f"TV: {tv_loss.item():.2f}"
            )

    # 9. Opsiyonel alpha blending
    if config['use_alpha_blending']:
        print(
            "Bilgi: Alpha blending uygulanıyor. "
            "Bu adım, stil aktarımı sonrası görüntünün orijinal fotoğraf yapısını "
            "daha fazla koruması için basit bir post-processing yaklaşımıdır."
        )
        target = blend_images(target, content, alpha=config['alpha'])

    # 10. Çıktıyı kaydet
    save_path = os.path.join(config['output_dir'], f"{config['label']}_output.jpg")
    save_image(target, save_path)

    print(f"Çıktı başarıyla kaydedildi: {save_path}")

    return save_path


def main():
    print("--- Deep Photo Style Transfer (Basitleştirilmiş Versiyon) ---")
    print(
        "Akademik Not: Bu proje, Deep Photo Style Transfer makalesinden esinlenmiştir. "
        "Makaledeki orijinal yöntem photorealism regularization için Matting Laplacian ve "
        "semantic segmentation kullanmaktadır. Bu projede ise daha uygulanabilir bir öğrenci "
        "projesi kapsamında VGG19 tabanlı Neural Style Transfer uygulanmış, fotoğraf gerçekçiliğini "
        "artırmak için total variation loss ve opsiyonel content-output blending kullanılmıştır.\n"
    )

    # ---------------------------------------------------------
    # 1. ÇALIŞTIRILACAK GÖRSEL ÇİFTLERİ
    # ---------------------------------------------------------
    image_pairs = [
        {
            "pair_name": "lake_sunset",
            "content_path": "images/content/lake_day.jpg",
            "style_path": "images/style/lake_sunset.jpg"
        },
        {
            "pair_name": "city_night",
            "content_path": "images/content/city_day.jpg",
            "style_path": "images/style/city_night.jpg"
        },
        {
            "pair_name": "forest_autumn",
            "content_path": "images/content/forest_green.jpg",
            "style_path": "images/style/forest_autumn.jpg"
        },
        {
            "pair_name": "mountain_foggy",
            "content_path": "images/content/mountain_day.jpg",
            "style_path": "images/style/mountain_foggy.jpg"
        },
        {
            "pair_name": "road_rainy",
            "content_path": "images/content/road_clear.jpg",
            "style_path": "images/style/road_rainy.jpg"
        }
    ]

    # ---------------------------------------------------------
    # 2. ORTAK PARAMETRELER
    # ---------------------------------------------------------
    base_config = {
        'image_size': 400,          # Bilgisayar zorlanırsa 300 yap
        'num_steps': 500,           # Test için 200 yapabilirsin
        'learning_rate': 0.03,
        'content_weight': 1,
        'tv_weight': 1e-4,
        'use_alpha_blending': True,
        'alpha': 0.85
    }

    # ---------------------------------------------------------
    # 3. HER GÖRSEL ÇİFTİ İÇİN 3 FARKLI STYLE WEIGHT DENEYİ
    # ---------------------------------------------------------
    experiments = [
        {
            'label': '1_soft',
            'style_weight': 1e5
        },
        {
            'label': '2_medium',
            'style_weight': 1e9
        },
        {
            'label': '3_extreme',
            'style_weight': 1e13
        }
    ]

    # outputs klasörü yoksa oluştur
    os.makedirs("outputs", exist_ok=True)

    # ---------------------------------------------------------
    # 4. TÜM GÖRSEL ÇİFTLERİNİ SIRAYLA ÇALIŞTIR
    # ---------------------------------------------------------
    for pair in image_pairs:
        pair_name = pair["pair_name"]
        content_path = pair["content_path"]
        style_path = pair["style_path"]

        print("\n========================================")
        print(f"Görsel çifti çalıştırılıyor: {pair_name}")
        print(f"Content: {content_path}")
        print(f"Style  : {style_path}")
        print("========================================")

        # Görüntü dosyalarının kontrolü
        if not os.path.exists(content_path):
            print(f"UYARI: '{content_path}' bulunamadı. Bu çift atlanıyor.")
            continue

        if not os.path.exists(style_path):
            print(f"UYARI: '{style_path}' bulunamadı. Bu çift atlanıyor.")
            continue

        # Her görsel çifti için ayrı çıktı klasörü oluştur
        output_dir = f"outputs/{pair_name}_results"
        os.makedirs(output_dir, exist_ok=True)

        results = []

        # Aynı görsel çifti için 3 farklı style_weight denemesi
        for exp in experiments:
            config = {
                **base_config,
                **exp,
                'content_path': content_path,
                'style_path': style_path,
                'output_dir': output_dir,
                'label': f"{pair_name}_{exp['label']}"
            }

            output_path = run_experiment(config)
            results.append((config['label'], output_path))

        # Her çift için ayrı comparison görseli oluştur
        comparison_path = f"outputs/{pair_name}_comparison.png"
        create_comparison_plot(content_path, style_path, results, comparison_path)

        print(f"\n{pair_name} için karşılaştırma görseli kaydedildi: {comparison_path}")

    print("\n[BAŞARILI] Tüm görsel çiftleri sırayla çalıştırıldı.")
    print("Oluşturulan tüm sonuçlara 'outputs/' klasöründen ulaşabilirsiniz.")


if __name__ == "__main__":
    main()