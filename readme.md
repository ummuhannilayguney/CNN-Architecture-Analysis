<div align="center">

# 🧠 YZM0206 Laboratuvar 6 Raporu: Evrişimli Sinir Ağları (CNN)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras&logoColor=white)](https://keras.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

*Bursa Teknik Üniversitesi - Bilgisayar Mühendisliği Bölümü*  
**Hazırlayan:** Ümmühan Nilay GÜNEY

</div>

---

## 📚 İçindekiler
1. [ADIM 1: Veri Kümesi Anatomisi ve Normalizasyon](#-adim-1-veri-kümesi-anatomisi-ve-normalizasyon)
2. [ADIM 2: Conv2D ve Padding Stratejileri (Aritmetik Analiz)](#-adim-2-conv2d-ve-padding-stratejileri-aritmetik-analiz)
3. [ADIM 3: Pooling vs. Stride 2 (Bilgi Azaltma)](#-adim-3-pooling-vs-stride-2-bilgi-azaltma)
4. [ADIM 4: Uçtan Uca Sınıflandırma (Fashion-MNIST)](#-adim-4-uçtan-uca-sınıflandırma-fashion-mnist)
5. [ADIM 5: Değerlendirme ve Raporlama (Ekstra Görev)](#-adim-5-değerlendirme-ve-raporlama-ekstra-görev)

---

## 📊 ADIM 1: Veri Kümesi Anatomisi ve Normalizasyon

### 1.1. Kanal Farkının Toplam Parametre Sayısına Matematiksel Etkisi
Bir `Conv2D` katmanındaki toplam parametre sayısını veren matematiksel formül şu şekildedir:

$$Parametre\ Sayısı = (Filtre\ Genişliği \times Filtre\ Yüksekliği \times Giriş\ Kanal\ Sayısı + 1) \times Filtre\ Sayısı$$

> 💡 **Not:** *Buradaki `+1` ifadesi, her bir filtre için eklenen bias (sapma) değerini temsil eder.*

Aynı boyutta bir kernel (örneğin $3 \times 3$) ve aynı sayıda filtre (örneğin $32$) kullandığımız bir senaryoyu ele alalım:

- 🔹 **MNIST İçin (28x28x1):** Giriş kanal sayısı $1$'dir.
  $$Parametre\ Sayısı = (3 \times 3 \times 1 + 1) \times 32 = (9 + 1) \times 32 = 320$$
  
- 🔹 **CIFAR-10 İçin (32x32x3):** Giriş kanal sayısı (RGB renkli olduğu için) $3$'tür.
  $$Parametre\ Sayısı = (3 \times 3 \times 3 + 1) \times 32 = (27 + 1) \times 32 = 896$$

**📌 Sonuç:** Görüldüğü üzere, giriş görüntüsünün kanal sayısının artması, kullanılan evrişim filtresinin derinliğini de zorunlu olarak artırır. Bu durum, `Conv2D` katmanındaki toplam parametre sayısını giriş kanal sayısıyla doğru orantılı olarak katlar (bias hariç). Bu artış, modelin kapasitesini, bellekte kapladığı alanı, eğitim süresini ve **aşırı öğrenme (overfitting)** ihtimalini doğrudan etkileyen kritik bir faktördür.

### 1.2. Normalizasyonun Eğitim Stabilitesine ve Gradyanlara Etkisi
Görüntü verilerinde piksel değerleri $0$ ile $255$ arasında değişen tam sayılardır. Bu değerleri doğrudan sinir ağına beslemek birtakım optimizasyon zorluklarına yol açar. Değerleri $255$'e bölerek $[0, 1]$ aralığına çekmenin (Normalizasyon) temel faydaları şunlardır:

1. **Gradyan Patlaması ve Kaybolması (Exploding / Vanishing Gradients) Sorununu Önleme:** Derin sinir ağlarında ileri ve özellikle geri yayılım (backpropagation) sırasında, büyük girdi değerleri ağırlıklarla çarpılıp üst üste eklendiğinde, gradyanların kontrolsüz bir şekilde büyümesine veya sıfıra çok yaklaşmasına yol açabilir. Verilerin standardize edilmesi, türev hesaplamalarının sağlıklı sınırlar içerisinde kalmasına katkıda bulunur.
2. **Eğitim Stabilitesi ve Daha Hızlı Yakınsama (Convergence):** Özelliklerin hepsi benzer ve küçük bir aralıkta olduğunda, kayıp fonksiyonunun (loss function) yüzeyi çok daha simetrik (küresel) bir yapıya bürünür. Bu sayede, **Gradyan İnişi (Gradient Descent)** algoritması minimum noktasına doğru çok daha düzgün ve salınım yapmadan ilerler. Bu durum modelin global minimumu bulma şansını yükseltir.

---

## 🧮 ADIM 2: Conv2D ve Padding Stratejileri (Aritmetik Analiz)

Evrişimli bir katmanın çıktı boyutunu (uzamsal genişlik ve yüksekliği) belirleyen temel formül:

$$Çıktı\ Boyutu = \lfloor \frac{Giriş\ Boyutu - Kernel\ Boyutu + (2 \times Padding)}{Stride} \rfloor + 1$$

Aşağıda verilen senaryoların manuel olarak adım adım hesaplaması gösterilmiştir:

### 1. Durum: 28x28x1 Giriş, 3x3 Kernel, Stride 1, Padding 'valid'
- `'valid'` padding stratejisinde ekstra bir dolgu (sıfır dolgu) eklenmez, yani $Padding = 0$.
- **İşlem:** $Çıktı = \lfloor \frac{28 - 3 + (2 \times 0)}{1} \rfloor + 1 = \lfloor 25 \rfloor + 1 = 26$
- **Beklenen Çıktı Boyutu:** `26 x 26 x (Filtre Sayısı)`

### 2. Durum: 28x28x1 Giriş, 3x3 Kernel, Stride 1, Padding 'same'
- `'same'` padding stratejisinde amaç, giriş ve çıktı boyutlarını aynı tutmaktır. Kernel $3\times3$ olduğunda her bir kenara $1$ piksellik sıfır dolgusu yapılır, yani $Padding = 1$.
- **İşlem:** $Çıktı = \lfloor \frac{28 - 3 + (2 \times 1)}{1} \rfloor + 1 = \lfloor \frac{27}{1} \rfloor + 1 = 28$
- **Beklenen Çıktı Boyutu:** `28 x 28 x (Filtre Sayısı)`

### 3. Durum: 32x32x3 Giriş, 5x5 Kernel, Stride 2, Padding 'valid'
- `'valid'` padding, sıfır dolgu yok, $Padding = 0$. Adım boyu (Stride) $2$.
- **İşlem:** $Çıktı = \lfloor \frac{32 - 5 + (2 \times 0)}{2} \rfloor + 1 = \lfloor \frac{27}{2} \rfloor + 1 = \lfloor 13.5 \rfloor + 1 = 13 + 1 = 14$
- **Beklenen Çıktı Boyutu:** `14 x 14 x (Filtre Sayısı)`

<div align="center">
  <b>Model Summary Doğrulamaları</b><br>
  <img src="./Ekran görüntüsü 2026-05-07 134342.png" alt="Model Summary" width="600"/>
</div>

---

## 📉 ADIM 3: Pooling vs. Stride 2 (Bilgi Azaltma)

Derin öğrenmede evrişimsel özellikleri altörneklemek (downsampling) ve uzamsal boyutu küçültmek için iki yaygın yaklaşım vardır:

### 1. MaxPooling'in "Değişmezlik" (Invariance) Özelliği
`MaxPooling2D`, belirli bir pencere (örneğin 2x2) içerisindeki piksellerden sadece en yüksek aktivasyona sahip olanı alır ve geri kalanları yok sayar.
- ✅ **Avantajı:** Modelin yerel küçük ötelemelere (translation invariance) karşı çok güçlü ve dirençli olmasını sağlar. Öğrenilebilir parametre içermediğinden hesaplama/parametre maliyeti yoktur.
- ❌ **Dezavantajı:** Matristeki diğer piksellerin (bilginin) atılması sebebiyle sert bir bilgi kaybına yol açar.

### 2. Stride=2'nin "Öğrenilebilir Altörnekleme" (Learnable Downsampling) Özelliği
Havuzlama kullanmak yerine `Conv2D` katmanının `strides` parametresini $2$ yapmak, boyutu havuzlamayla aynı oranda küçültür ancak bunu evrişim operasyonunun ağırlıkları aracılığıyla yapar.
- ✅ **Avantajı:** Sabit bir kural uygulamak yerine ağın hangi bilgiyi koruyup hangisini eleyeceğini veriden yola çıkarak kendisinin öğrenmesini sağlar (ResNet gibi mimarilerde esnekliği artırır).
- ❌ **Dezavantajı:** Hesaplama karmaşıklığını ve parametre sayısını artırır.

> 🏆 **Karar (Ne Zaman Hangisi Avantajlıdır?)**  
> Modelin parametre sayısının düşük tutulması gereken basit problemler için **MaxPooling** avantajlıdır. Ağın kapasitesinin yüksek olduğu, büyük veri setleriyle çalışılan uçtan uca sistemlerde ise **Stride=2** stratejisi daha avantajlıdır.

<div align="center">
  <img src="./Ekran görüntüsü 2026-05-07 133459.png" alt="Pooling vs Stride" width="600"/>
</div>

---

## 👕 ADIM 4: Uçtan Uca Sınıflandırma (Fashion-MNIST)

### 4.1. Mimari Kurulum ve Eğitim Çıktıları
Aşağıda, kurulan mimari ve eğitim sürecine ait log çıktıları bulunmaktadır:

<div align="center">
  <img src="./image_e0e6ac.jpg" alt="Mimari Kod" width="600"/>
  <br>
  <img src="./image_e0e3c4.jpg" alt="Eğitim Logu" width="600"/>
</div>

### 4.2. Hata Matrisi ve Sınıflandırma Analizi

<div align="center">
  <img src="./Ekran görüntüsü 2026-05-07 134000.jpg" alt="Hata Matrisi" width="600"/>
</div>

Fashion-MNIST eğitimleri sonrası hata matrisinde (confusion matrix) gözlemlenen en tipik karışıklıklar şunlardır:
1. **T-shirt/top** ile **Shirt (Gömlek)**
2. **Pullover (Kazak)** ile **Coat (Kaban)**
3. **Sneaker (Spor Ayakkabı)** ile **Ankle boot (Bilekte Bot)**

**🔍 Teknik Analiz: Neden T-shirt ve Gömlek karıştırılıyor?**  
Evrişimli Sinir Ağları (CNN), karar sınırlarını belirlerken nesnelerin "dış hatları", "kenar yoğunlukları" ve "köşe dizilimleri" gibi uzamsal özelliklerine (feature extraction) güvenir. 
- **Görsel Silüet:** "T-shirt" ve "Shirt" şekil itibarıyla birbirlerine son derece benzerdir (kısa kol, yaka açıklığı, gövde uzunluğu). CNN'in alt katmanları bu iki sınıf için neredeyse aynı silüet çıktılarını üretir.
- **Çözünürlük Engeli:** Görüntüler sadece $28\times28$ piksel ve gri tonlamalıdır. Bir gömlek ile T-shirt arasındaki belirleyici farklar ince detaylarda (yaka kesimi, düğme dizilimi) yatar. Düşük çözünürlükte bu **ince taneli özellikler (fine-grained features)** kaybolduğu için ağ ayrım yapmakta zorlanır.

---

## 📈 ADIM 5: Değerlendirme ve Raporlama (Ekstra Görev)

Üç farklı veri setinde ortak mimari ile elde edilen başarı metrikleri aşağıdaki gibidir:

<div align="center">
  <img src="./image_e090b4.jpg" alt="Ortak Kod" width="600"/>
  <br>
  <img src="./Ekran görüntüsü 2026-05-07 134345.png" alt="Test Doğrulukları" width="500"/>
</div>

### Kıyaslama ve Yorumlar
Elde edilen **MNIST (%97.6)**, **Fashion-MNIST (%88.5)** ve **CIFAR-10 (%57.6)** test doğruluk oranları arasındaki belirgin varyasyonların teknik sebepleri şu şekildedir:

1. 🌟 **MNIST (%97.6 Doğruluk):**
   - **Karmaşıklık:** Son derece düşüktür. Tamamen siyah arka plan üzerinde yüksek kontrastlı rakamlardan oluşur.
   - **Uzamsal Yapı:** $28\times28$ boyutlu ve tek kanallıdır (siyah-beyaz). Uzamsal gürültü minimum seviyede olduğundan çok sığ bir CNN bile kenar özelliklerini rahatça çıkararak problemi mükemmele yakın şekilde çözer.

2. 👗 **Fashion-MNIST (%88.5 Doğruluk):**
   - **Karmaşıklık:** Orta seviyededir. Çözünürlük MNIST ile aynı olmasına rağmen kıyafetlerin morfolojik varyasyonları (açılar, duruşlar) fazladır.
   - **Uzamsal Yapı:** Sınıflar arası (T-shirt/Shirt gibi) aşırı silüet benzerlikleri, problemin özellik karmaşıklığını artırmıştır. Modelin detayları ayrıştırabilmesi için daha fazla kapasiteye ihtiyacı vardır, bu yüzden başarı MNIST'e göre düşer.

3. 🚗 **CIFAR-10 (%57.6 Doğruluk):**
   - **Karmaşıklık:** Çok yüksektir. Gerçek dünyadan çekilmiş nesne fotoğraflarından oluşur.
   - **Uzamsal Yapı:** Görüntüler renkli ($32\times32\times3$), arka planlar ise gürültülüdür. Nesneler ölçek, ışık ve duruş olarak çok büyük farklılıklar gösterir. Bu dağınık veriden uzamsal özellikleri öğrenebilmek sığ CNN'ler için oldukça zordur. Bu problemi çözebilmek için ResNet veya VGG gibi daha derin mimarilere ve **Veri Artırma (Data Augmentation)** tekniklerine ihtiyaç vardır.

---
*Bu rapor, derin öğrenme temellerini anlamak ve Keras kullanarak Evrişimli Sinir Ağlarının anatomisini incelemek amacıyla hazırlanmıştır.*
