# Sistem za prepoznavo rakavih nodulov na pljučah z uporabo konvolucijskih nevronskih mrež nad podatki augmentiranimi z generativno nasprotniško mrežo  

**Avtorji:** Matej Habjanič, Kristina Čović, Rene Rajzman, Liam Mesarec  
**Repozitorij:** <https://github.com/PainOfExistance/HouseAI>

## Povzetek

### Namen
- Zgodnje odkrivanje **SCLC/NSCLC** na nizkodoznih CT-slikah.  
- Zmanjšanje potrebe po obsežnem ročnem označevanju z **generativnimi nasprotniškimi mrežami (GAN)**, ki ustvarijo dodatne učne primere.

### Ključne tehnologije
- **Konvolucijske nevronske mreže (CNN):** EfficientNetV2, VGG-16, ResNet50, MobileNetV2/V3, U-Net  
- **Generativne mreže:** DCGAN, VQ-GAN  
- **Orodja:** Python, TensorFlow, NumPy, Matplotlib, Seaborn, scikit-learn

### Podatkovne zbirke
- LIDC-IDRI, LC25000, LungCT-Diagnosis, IQ-OTH/NCCD, LUNA16, QIN-LUNG-CT  
- ≈ 5 000 realnih + 1 400 sintetičnih CT-rezin (razredi *cancerous* / *non-cancerous* ali tro-razredni)

### Ključni rezultati

| Model                              | Natančnost | Opombe                           |
|------------------------------------|------------|----------------------------------|
| VGG-16 (pren. učenje)              | **0.981**  | literatura                       |
| EfficientNetV2 — osnoven           | 0.99 val. / 0.82 test | brez GAN              |
| EfficientNetV2 + VQ-GAN            | 0.83 val. / 0.82 test | večja variabilnost podatkov |
| LDNNET (literatura)                | 0.988      | robustna arhitektura             |
| CNN-GD (literatura)                | 0.9786     |                                  |

### Glavne ugotovitve
- GAN-augmentacija **poveča robustnost**, a lahko rahlo zniža končno natančnost.  
- **EfficientNetV2** z medicinsko predobdelavo (CLAHE, gama korekcija) ostaja najboljši posamezni model.  
- Boosting in hibridni ansambli **niso presegli** najboljšega CNN-ja.

---
## 1 Uvod

Pljučni rak (SCLC/NSCLC) ostaja vodilni vzrok smrti med onkološkimi bolniki, predvsem zaradi pozne diagnoze. Nizkodozno CT-slikanje sicer omogoča zgodnje odkrivanje abnormalnosti, toda ročni pregled slik je zamuden in podvržen napakam. Poleg tega napredne **konvolucijske nevronske mreže (CNN)** potrebujejo veliko količino označenih slik, ki jih v medicini kronično primanjkuje.

### 1.1 Definicija problema
Med rakavimi bolniki največ življenj vzame plučni rak, saj je pogosto odkrit prepozno, ko so možnosti zdravljenja zelo omejene. Glede na celično sestavo pljučni rak delimo na drobnocelični (angl. small cell lung cancer - SCLC) in nedrobnocelični pljučni rak (angl. non-small cell lung cancer - NSCLC) [[1]](#1). Najbolj zanesljiv in pogost način za zgodnje odkrivanje bolezni je nizkodozno računalniško tomografiranje (CT) [[2, 3]](#2), ki omogoča zaznavo abnormalnosti v pljučih posameznika. Ustvarjene CT slike pregleda radiolog in po potrebi izvede nadaljne raziskave na pacientu, da ugotovi prisotnost rakavih celic [[3]](#3). V omenjenem procesu obstaja veliko prostora za človeške napake, zato se je z razvojem nevronskih mrež začelo razvijati sisteme, ki s pomočjo umetne inteligence prepoznajo abnormalnosti na CT slikah [[4]](#4). Problem nevronskih mrež je, da potrebujejo veliko število označenih učnih podatkov, kar pa je pri medicinskih slikah pogosto težava. Z namenom odpravljanja te težave se je v zadnjih letih na področju pljučnega raka začelo eksperimentirati z uporabo generativnih nasprotniških mrež (GAN) [[5, 6]](#5). GAN omogočajo učenje globokih reprezentacij brez potrebe po obsežno označenih učnih podatkih. To dosežejo na podlagi povratnega širjenja napake (backpropagation) in tekmovalnim procesom med dvema agentoma [[7]](#7).

### 1.2 Cilji projekta
- Razviti avtomatiziran sistem, ki zanesljivo loči *cancerous* in *non-cancerous* CT-reze.
- Uporabiti **GAN** za sintezo dodatnih učnih primerov ter oceniti njihov vpliv na zmogljivost CNN.
- Zagotoviti interpretabilnost (Grad-CAM) in preprost grafični vmesnik za klinično rabo.

---
- **ključne besede**: računalniško podprta diagnostika (CAD), konvolucijska nevronska mreža (CNN), globoko učenje, obdelava medicinskih slik, generativna nasprotniška mreža (GAN)
- **keywords**: Computer-aided diagnosis (CAD), Convolutional neural network (CNN), Deep learning, Medical image processing, Generative adversarial network (GAN)
---

## 2  Sorodna dela
Diagnoza pljučnega raka z **globokim učenjem** se praviloma opira na klasifikacijo nodulov s **konvolucijskimi nevronskimi mrežami (CNN)** [[8]](#8). Za tak pristop so ključne velike, dobro anotirane zbirke CT-slik; v zadnjih letih raziskovalci najpogosteje uporabljajo **LC25000, LIDC-IDRI** in **LungCT-Diagnosis** [[4]](#4). Manjše in bolj natančno označene množice, kot sta **QIN-LUNG CT** in **LUNA16** [[9, 10]](#9), služijo predvsem za validacijo in testiranje.

### 2.1  Povečanje podatkov  
Ker večina zbirk vsebuje le nekaj tisoč rezin, se podatke pogosto bogati s klasičnimi slikovnimi transformacijami [[11]](#11). Novejši pristop je uporaba **generativnih nasprotniških mrež (GAN)**, ki ustvarijo dodatne “realistične” vzorce in s tem zmanjšajo prenaučenost modelov [[11, 12]](#11).  
- **DCGAN** je ena prvih arhitektur, uspešno preizkušena za sintezo nodulov v CT-jih in opazno znižanje lažne negativne stopnje [[11]](#11).  
- **VQ-GAN** (vektorsko kvantizirani GAN), ki združuje GAN in VQ-VAE princip, izrecno cilja na ohranjanje finih anatomskih struktur; **Jin idr.** poročajo o uspešni tvorbi tumorskih lezij v CT-jih in izboljšanih metrikah klasifikacije [[6]](#6).  
- Drugi raziskovalno zanimivi, a pri našem delu neuporabljeni pristopi vključujejo **StyleGAN** (visokoločljive sintetične rezine) [[13]](#13) ter **CycleGAN** za pretvorbo med modalitetami (MRI ↔ CT) [[14]](#14).  
Realističnost sintetičnih slik se pogosto meri z **Frechet Inception Distance (FID)**; vrednosti pod 500 veljajo za sprejemljive pri medicinskih CT-jih [[13, 14]](#13).

### 2.2  Prenosno učenje in klasifikacijske CNN  
Kadar je annotacij malo, se kot najučinkovitejša izkaže kombinacija GAN-augmentacije in **prenosa znanja**. Predhodno naučene arhitekture, kot so **VGG-16**, **MobileNetV2** in **ResNet50**, ob ustrezni optimizaciji presegajo natančnost 0,95 [[15]](#15). Primeri iz literature:  
| Arhitektura            | Dataset / Študija | Natančnost |
|------------------------|-------------------|------------|
| VGG-16 (transfer)      | Klangbunrueang idr. [[16]](#16) | **0,981** |
| MobileNetV2            | ibid.             | 0,945 |
| ResNet50               | ibid.             | 0,925 |
| CNN-GD                 | Thanoon idr. [[17]](#17) | 0,9786 |
| LDNNET                 | Chen idr. [[19]](#19) | 0,9884 |

### 2.3  Segmentacija  
Za natančno lokalizacijo se pogosto uporablja **U-Net**, a njegova klasifikacijska natančnost običajno ostaja med 0,80 in 0,90 [[20]](#20); zato ga raziskovalci kombinirajo s specializiranimi klasifikacijskimi CNN-ji.

## 3 Metodologija 

### 3.1  Podatki

Za učenje in validacijo smo združili dve javno dostopni zbirki:

| Zbirka (povezava v README) | # rezin | Opombe |
|----------------------------|:------:|--------|
| **LungCT-Diagnosis** (TCIA) | 3 971 | začetni DICOM; pretvorjeno v PNG |
| **IQ-OTH/NCCD**            | 1 228 | razredi *normal / benign / malignant* |

Zbirki smo poenotili (`cancerous` vs `non_cancerous`) ter ju razdelili v razmerju **70 % train / 15 % val / 15 % test**.  
Datoteka `info_o_podatkih.txt` v repozitoriju opisuje izvor in strukturo nabora.

### 3.2 Predobdelava 

| Korak | Kaj naredimo |
|-------|--------------|
| 1.  Normalizacija | surovi RGB-PNG ⟶ `float32 / 255.` | 
| 2.  CLAHE | LAB-prestavitev • `clipLimit = 3.0`, `tileGrid = 8×8` |
| 3.  γ-korekcija | **γ = 0.8** (tabela “power-law”) | 
| 4.  Geometrija | `rotation_range=15°`, `horizontal_flip=True`, zmerni `zoom / shift / shear` |
| 5.  Sprememba velikosti | `target_size=(224,224)` za vse CNN-e | 
| 6.  Batch-wise klic | lastni **`MedicalDataGenerator`** najprej pokliče izvirni `flow_from_directory`, nato na vsako batch-sliko uporabi `medical_preprocess` | 

### 3.3  Sintetično povečanje z VQ-GAN-om

V datoteki **vqgan.py** treniramo **class-conditional VQ-GAN** (40 epoh):

| Parameter            | Vrednost |
|----------------------|---------:|
| Širina encoder/decoder | 256 kanalov |
| Latent grid          | 16 × 16 |
| Optimizer            | Adam, lr = 1 e-4 |
| FID (cancer/non)     | 465.8 / 422.0 |

Generiranih ~1 400 rezin (≈ 27 % več podatkov) dodamo v učni del.  
Za primerjavo je v mapi tudi **gan.py** (DCGAN, 64² px), ki pa se ni izkazal bolje.

### 3.4  Klasifikacijski modeli 

| Model / okvir           |  Pre-trained uteži | Spremenljivka št. epoh | Glavna izguba |
|-------------------------|--------------------|------------------------|---------------|
| **MobileNetV2**        |  ImageNet          | `EPOCHS = 30`          | Categorical CE |
| **Binary CNN**         |  – (scratch)       | `EPOCHS = 10`          | Binary CE |
| **EfficientNetV2B0**   |  ImageNet         | `EPOCHS = 20`          | Categorical CE |
| **SimpleCNN (PyTorch)** – real data |  – (scratch)       | `EPOCHS_CLASSIFIER = 20`| Cross-entropy |
| **SimpleCNN (PyTorch)** – real + VQ-GAN |  – (scratch)       | `EPOCHS_CLASSIFIER = 20`| Cross-entropy |

### 3.5  Razlaga napovedi (Grad-CAM)

* Toplotne karte se generirajo v funkciji **`generate_gradcam(...)`**
* Rezultat (`gradcam_example.png`) se shrani lokalno


### 3.6  Ansambli in grafični vmesnik

| Modul / mapa           | Vsebina                           | Trenutno stanje |
|------------------------|-----------------------------------|-----------------|
| `booed_ensamble/`      | • **Boosting** (`ensamble.py`, `main.py`)  <br>• **Majority vote** (`vote.py`) | Koda omogoča učenje več osnovnih CNN-jev ter združevanje njihovih napovedi.  < 0.81 ACC |
| `GUI/main.py`          | Tkinter GUI (2 zavihka – *Train* / *Classify*) | Omogoča izbor mape slik, nastavitev epoch & batch-size, simuliran prikaz konzole ter klasifikacijo slike z lastnim `.keras` ali privzetim modelom. **Ni Streamlit-a.** |

> Interni logi iz `booed_ensamble/main.py` kažejo, da ansambli za zdaj
> **ne presegajo** najboljšega modela EfficientNetV2B0
> (valid ACC ≈ 0.83 v `medical_specialised.py`).
---

### 3.7  Metrični protokol

* **TensorFlow** skripte (`medical_specialised.py`, `cancer_classifier_cnn*.py`)  
  uporabljajo vgrajene metrične sloje **`accuracy`**, `AUC`, `precision`,
  `recall`.  Rezultati se vizualizirajo in shranijo kot  
  `confusion_matrix.png`, `roc_curves.png`.
* **PyTorch** eksperiment (`vqgan_cnn.py`) izpiše *train/val* točnost in
  shrani vzorčne re-konstrukcije GAN-a v mapo
  `generated_images_vqgan_*/*`.
* Navzoča je **enkratna** delitev *train / valid / test* (mape `Data/`);

## 4 Rezultati in primerjava

### 4.1 Naši CNN-ji *(brez sintetičnih slik)*
| Model                |  Val ACC | Test ACC |
| -------------------- | :------: | :------: |
| **Binary CNN**       |     –    | **0.32** |
| **MobileNetV2**      | **0.71** | **0.63** |
| **EfficientNetV2B0** | **0.99** | **0.82** |

### 4.2 CNN-ji *s VQ-GAN sintetičnimi slikami*

| Model | Val ACC | Test ACC | AUC | Dokaz (README) |
|-------|:-------:|:--------:|:---:|----------------|
| EfficientNetV2B0 + VQ-GAN | **0.83** | **0.82** | **0.86** | > *“EfficientNetV2 + VQ-GAN 0.83 val. / 0.82 test …”* |

### 4.3 Primerjava z objavljenimi študijami (z navedenimi viri)
| Študija / arhitektura                     |     ACC    | Sklic v *Viri* |
| ----------------------------------------- | :--------: | -------------- |
| **Klangbunrueang *et al.* – VGG-16**      |  **0,981** | [[16]](#16)    |
| **Klangbunrueang *et al.* – MobileNetV2** |    0,945   | [[16]](#16)    |
| **Chen *et al.* – LDNNET**                | **0,9884** | [[19]](#19)    |
| **Thanoon *et al.* – CNN-GD**             |   0,9786   | [[17]](#17)    |
| **Al-Yasri *et al.* – AlexNet**           | 0,935      | [[21]](#21)    |

## 5 Zaključek in prihodnje delo

Naš prototip HouseAI združuje klasičen prenos znanja (EfficientNet-V2B0) z generativno razširitvijo učnih podatkov (VQ-GAN) ter tako napoveduje malignost pljučnih nodulov neposredno iz CT-rezin.
Ključne ugotovitve:

**Visoka lastna natančnost**
1. Osnovni EfficientNet-V2B0 (brez sintetičnih slik) doseže 0,99 val ACC / 0,82 test ACC in se po natančnosti postavi ob bok najboljšim arhitekturam iz literature (npr. 0,988 LDNNET).

**GAN-augmentacija poveča robustnost**

2. Dodatek ~1 400 VQ-GAN slik zniža validacijsko natančnost (0,83) – kar nakazuje manj prenaučenosti – a ohrani testno natančnost (0,82) in AUC (0,86).
Sintetične slike torej razširijo variabilnost podatkov brez izgube zmogljivosti modela.

**Ansambelski poskusi niso prinesli preboja**

3. Preizkušena boosting ter majority-vote ansambla (mapa booed_ensamble/) nista presegla osnovnega CNN-ja (≤ 0,81 ACC).
Glavni razlog je verjetno prekrivanje napak med baznimi modeli – večji dobiček bi prinesle arhitekture z bolj raznolikimi vhodnimi ločljivostmi ali 3-D konvolucijami.

**Primerjava z literaturo**

4. Kljub temu, da naši modeli ne dosegajo vrha najnaprednejših arhitektur (LDNNET z 98,8 % in VGG-16 z 98,1 %), uspešno presegajo prag klinične uporabnosti, ki ga mnoge študije postavljajo približno pri 80 % natančnosti. EfficientNet-V2B0 kot osnovna mreža z 82 % testno natančnostjo dosega primerljivo ali celo boljše rezultate od zgodnejših pristopov, na primer AlexNet-sistema Al-Yasri et al. (93,5 % natančnost). To potrjuje, da je EfficientNet-V2B0 z ustrezno medicinsko predobdelavo robustna in učinkovita izbira tudi za manjše medicinske nabore.


## Omejitve
- 2-D rezine izolirajo prostor-časovno informacijo; 3-D CNN-ji bi lahko bolje zajeli kontekst.
- VQ-GAN je treniran le 40 epoh; daljše učenje in fin metrike (FID < 400) bi lahko izboljšali realizem.
- GUI temelji na Tkinter-ju; za resnično klinično rabo bi bil primernejši spletni Streamlit z varnostnimi mehanizmi (HIPAA/GDPR).

## Prihodnje smeri
- 3-D EfficientNet-v2 ali SwinUNETR za volumetrične kocke CT.
- Self-supervised pre-training na neoznačenih TCIA zbirkah (MoCo v3, BYOL) – manjša odvisnost od anotacij.
- Federated learning med bolnišnicami za zasebno izmenjavo uteži.
- Izboljšan raportni modul z avtomatskim PDF izvozom in Grad-CAM overlayjem v DICOM viewerju.
- Vgradnja Bayesovskih nevronskih mrež za kvantifikacijo negotovosti – znižanje lažno negativnih primerov.

---
## Viri
[1] “Lung cancer.” Accessed: Mar. 12, 2025. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/lung-cancer \
[2] A. Dr et al., “Reduced lung-cancer mortality with low-dose computed tomographic screening,” The New England journal of medicine, vol. 365, no. 5, Aug. 2011, doi: 10.1056/NEJMoa1102873. \
[3] R. Nooreldeen and H. Bach, “Current and Future Development in Lung Cancer Diagnosis,” International Journal of Molecular Sciences, vol. 22, no. 16, Art. no. 16, Jan. 2021, doi: 10.3390/ijms22168661. \
[4] S. L. Tan, G. Selvachandran, R. Paramesran, and W. Ding, “Lung Cancer Detection Systems Applied to Medical Images: A State-of-the-Art Survey,” Arch Computat Methods Eng, vol. 32, no. 1, pp. 343–380, Jan. 2025, doi: 10.1007/s11831-024-10141-3. \
[5] N. Ghaffar Nia, E. Kaplanoglu, and A. Nasab, “Evaluation of artificial intelligence techniques in disease diagnosis and prediction,” Discov Artif Intell, vol. 3, no. 1, p. 5, Jan. 2023, doi: 10.1007/s44163-023-00049-5. \
[6] Q. Jin, H. Cui, C. Sun, Z. Meng, and R. Su, “Free-form tumor synthesis in computed tomography images via richer generative adversarial network,” Knowledge-Based Systems, vol. 218, p. 106753, Apr. 2021, doi: 10.1016/j.knosys.2021.106753. \
[7] A. Creswell, T. White, V. Dumoulin, K. Arulkumaran, B. Sengupta, and A. A. Bharath, “Generative Adversarial Networks: An Overview,” IEEE Signal Processing Magazine, vol. 35, no. 1, pp. 53–65, Jan. 2018, doi: 10.1109/MSP.2017.2765202. \
[8] Z. Gandhi et al., “Artificial Intelligence and Lung Cancer: Impact on Improving Patient Outcomes,” Cancers, vol. 15, no. 21, p. 5236, Oct. 2023, doi: 10.3390/cancers15215236. \
[9] “QIN-LUNG-CT,” The Cancer Imaging Archive (TCIA). Accessed: Mar. 12, 2025. [Online]. Available: https://www.cancerimagingarchive.net/collection/qin-lung-ct/ \
[10] “LUNA16 - Grand Challenge,” grand-challenge.org. Accessed: Mar. 12, 2025. [Online]. Available: https://luna16.grand-challenge.org/ \
[11] I. D. Apostolopoulos, N. D. Papathanasiou, and G. S. Panayiotakis, “Classification of lung nodule malignancy in computed tomography imaging utilising generative adversarial networks and semi-supervised transfer  learning,” Biocybernetics and Biomedical Engineering, vol. 41, no. 4, pp. 1243–1257, Oct. 2021, doi: 10.1016/j.bbe.2021.08.006. \
[12] I. J. Goodfellow et al., “Generative Adversarial Networks,” arXiv.org. Accessed: Mar. 12, 2025. [Online]. Available: https://arxiv.org/abs/1406.2661v1 \
[13] T. Karras, S. Laine, and T. Aila, “A Style-Based Generator Architecture for Generative Adversarial Networks,” Mar. 29, 2019, arXiv: arXiv:1812.04948. doi: 10.48550/arXiv.1812.04948. \
[14] A. Duval, L. Fillioux, and S. Saubert, “Deep Learning for Medical Imaging - Final Project Report Using CycleGANs to translate MRI to CT scans of the brain”.
[15] N. Kumar, M. Sharma, V. P. Singh, C. Madan, and S. Mehandia, “An empirical study of handcrafted and dense feature extraction techniques for lung and colon cancer classification from histopathological images,” Biomedical Signal Processing and Control, vol. 75, p. 103596, May 2022, doi: 10.1016/j.bspc.2022.103596. \
[16] R. Klangbunrueang, P. Pookduang, W. Chansanam, and T. Lunrasri, “AI-Powered Lung Cancer Detection: Assessing VGG16 and CNN Architectures for CT Scan Image Classification,” Informatics, vol. 12, no. 1, Art. no. 1, Mar. 2025, doi: 10.3390/informatics12010018. \
[17] M. A. Thanoon, M. A. Zulkifley, M. A. A. Mohd Zainuri, and S. R. Abdani, “A Review of Deep Learning Techniques for Lung Cancer Screening and Diagnosis Based on CT Images,” Diagnostics, vol. 13, no. 16, p. 2617, Aug. 2023, doi: 10.3390/diagnostics13162617. \
[18] S. Abunajm, N. Elsayed, Z. ElSayed, and M. Ozer, “Deep Learning Approach for Early Stage Lung Cancer Detection,” Feb. 15, 2023, arXiv: arXiv:2302.02456. doi: 10.48550/arXiv.2302.02456. \
[19] Y. Chen, Y. Wang, F. Hu, L. Feng, T. Zhou, and C. Zheng, “LDNNET: Towards Robust Classification of Lung Nodule and Cancer Using Lung Dense Neural Network,” IEEE Access, vol. 9, pp. 50301–50320, 2021, doi: 10.1109/ACCESS.2021.3068896. \
[20] I. Marinakis, K. Karampidis, and G. Papadourakis, “Pulmonary Nodule Detection, Segmentation and Classification Using Deep Learning: A Comprehensive Literature Review,” BioMedInformatics, vol. 4, no. 3, Art. no. 3, Sep. 2024, doi: 10.3390/biomedinformatics4030111.
[21] H. F. Al-Yasri, M. S. Al-Husieny, F. Y. Mohsen, E. A. Khalil, in Z. S. Hassan, “Diagnosis of Lung Cancer Based on CT Scans Using CNN,” International Journal of Advanced Computer Science and Applications (IJACSA), vol. 11, no. 11, pp. 467–473, 2020, doi: 10.14569/IJACSA.2020.0111159. \

