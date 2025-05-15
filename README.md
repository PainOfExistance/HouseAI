
# Sistem za prepoznavo rakavih nodulov na pljučah z uporabo konvolucijskih nevronskih mrež nad podatki augmentiranimi z generativno nasprotniško mrežo

## Definicija problema
Med rakavimi bolniki največ življenj vzame plučni rak, saj je pogosto odkrit prepozno, ko so možnosti zdravljenja zelo omejene. Glede na celično sestavo pljučni rak delimo na drobnocelični (angl. small cell lung cancer - SCLC) in nedrobnocelični pljučni rak (angl. non-small cell lung cancer - NSCLC) [[1]](#1). Najbolj zanesljiv in pogost način za zgodnje odkrivanje bolezni je nizkodozno računalniško tomografiranje (CT) [[2, 3]](#2), ki omogoča zaznavo abnormalnosti v pljučih posameznika. Ustvarjene CT slike pregleda radiolog in po potrebi izvede nadaljne raziskave na pacientu, da ugotovi prisotnost rakavih celic [[3]](#3). V omenjenem procesu obstaja veliko prostora za človeške napake, zato se je z razvojem nevronskih mrež začelo razvijati sisteme, ki s pomočjo umetne inteligence prepoznajo abnormalnosti na CT slikah [[4]](#4). Problem nevronskih mrež je, da potrebujejo veliko število označenih učnih podatkov, kar pa je pri medicinskih slikah pogosto težava. Z namenom odpravljanja te težave se je v zadnjih letih na področju pljučnega raka začelo eksperimentirati z uporabo generativnih nasprotniških mrež (GAN) [[5, 6]](#5). GAN omogočajo učenje globokih reprezentacij brez potrebe po obsežno označenih učnih podatkih. To dosežejo na podlagi povratnega širjenja napake (backpropagation) in tekmovalnim procesom med dvema agentoma [[7]](#7).

- **ključne besede**: računalniško podprta diagnostika (CAD), konvolucijska nevronska mreža (CNN), globoko učenje, obdelava medicinskih slik, generativna nasprotniška mreža (GAN)
- **keywords**: Computer-aided diagnosis (CAD), Convolutional neural network (CNN), Deep learning, Medical image processing, Generative adversarial network (GAN)

## Pregled sorodnih del
Diagnoza pljučnega raka s pomočjo globokega učenja ali CNN temelji na klasificiranju škodljivosti pljučnih nodulov [[8]](#8). Za to potrebujemo zelo veliko množico učnih podatkov. Najpogosteje uporabljene v najboljših modelih v letih 2021 do 2024 so LC25000, LIDC-IDRI in LungCT-Diagnosis [[4]](#4). Veliko je tudi manjših množic kot so QIN LUNG CT in LUNA2016 [[9, 10]](#9). Ker so učne množice majhne s samo nekaj tisoč slikami pljučnih nodulov, so podatki pogosto augmentirani s slikovnimi transformacijami [[11]](#11). Novejša metoda augmentacije je GAN, ki namesto geometrijskega transformiranja tvori nove slike s tekmovanjem med dvema nevronskima mrežama kjer ena mreža generira podatke, druga pa poskuša ugibati če so ti podatki prišli iz originalne učne množice ali so bili generirani [[11, 12]](#11). StyleGAN generira nove slike na podlagi značilk učnih podatkov, kar bi lahko bilo uporabljeno za učenje značilk pljučnih nodulov [[13]](#13). CycleGAN lahko generira slike, ki so bile zajete v različnih domenah, kot je pretvorba MRI v CT [[14]](#14). Za preverjanje realističnosti slike se uporablja Fretchet Inception Distance [[13, 14]](#13). Problem majhnih množic učnih podatkov se lahko rešuje z uporabo predhodno naučene nevronske mreže, ki lahko dosegajo natančnost nad 0.95, vendar zahtevajo dobro optimizacijo z uporabo metod prenosa znanja [[15]](#15). Rezultati ene izmed študij kažejo, da je VGG-16 dosegel natančnost 0.981, MobileNet  0.945, ResNet50 0.925 in InceptionV3 0.92 [[16]](#16). Večjo natančnost so zasledili pri modelih CNN GD z 0.9786 [[17]](#17) in 0.9945 s CNN [[18]](#18). Zelo visoko natančnost ima tudi LDNNET z natančnostjo 0.988396 [[19]](#19). U-Net modeli so popularni za medicinske slike vendar imajo manjšo natančnost med 0.8 do 0.9 [[20]](#20). 

## Načrt rešitve
- **skupina:** 1
- **sodelavci:** Matej Habjanič, Kristina Čović, Rene Rajzman, Liam Mesarec
- **repozitorij:** https://github.com/PainOfExistance/HouseAI
- **programski jezik:** Python
- **opravila**
    1. razvojna iteracija
        - pregled podatkovnih zbirk in izbor katere bomo uporabili
        - vzpostavitev okolja za programiranje
        - priprava podatkov za uporabo s knjižnico Tensorflow
    2. razvojna iteracija
        - testiranje GAN s CT slikami
        - implementacija CNN
        - učenje podpornih osnovnih metod strojnega učenja
    3. razvojna iteracija
        - testiranje z drugimi CNN pred-trenirani modeli
        - optimizacija
        - gradnja hibridov iz nabora različnih klasifikacijskih metod
    4. razvojna iteracija
        - uporabniški vmesnik za aplikacijo
        - primerjava rezultatov z drugimi modeli
        - pisanje članka
- **UML diagram rešitve**:

![houseai](https://github.com/user-attachments/assets/bdb0191e-746e-43b0-a5b0-72d017388a71)


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

## 1. šprint
Liam Mesarec, Kristina Čović: V sklopu opravila "pregled podatkovnih zbirk in izbor katere bomo uporabili" sva šla čez podatkovne množice ki jih lahko uporabimo za strojno učenje, o njih sma napisala informacije o tem kaj zajemajo, kako so anotirani, v katerem formatu so slike, katere modifikacije so bile vnaprej narejene nad slikami, koliko podatkov je v množici in nad koliko pacientov so bili izvedeni in viri podatkov. Razvila sva načrt katere učne podatke bi lahko kombinirali in katere tipe rakov bi lahko upoštevali.

Matej Habjanič: V sklopu prvega sprinta je bila moja naloga pripraviti delovno okolje za našo delo. Zaradi uporabe programskega jezika Python, smo vspostavili venv okolje ter tudi izbrali knjižnice, ki jih bomo uporabljali. Te so sledeče:  
    - tensorflow: grajenje nevronskih mrež  
    - numpy: za hitre operacija in interoperabilnost s tensorflow  
    - matplotlib: vizualizacija podatkov  
    - seaborn: vizualizacija matrike zmede  
    - sklearn: predobdelava in kategoriziranje podatkov  
Po izdelavi preliminarnega dela sem tudi omogočil vizualizacijo natančnosti, izgibe in matreke zmede, kot tudi spreminjanje le teh skozi čas.  

Rene Rajzman:  V sklopu prvega sprinta sem razvil in treniral dva modela za klasifikacijo medicinskih slik. Prvi model je bila osnovna konvolucijska nevronska mreža za binarno klasifikacijo, ki je dosegla 32 % natančnost. Drugi model je temeljil na prenosnem učenju z MobileNetV2 in je dosegel 63 % natančnost. Oba modela sem shranil za nadaljnjo uporabo.

## 2. šprint
Liam Mesarec, Kristina Čović: V sklopu drugega šprinta sva se posvetila **testiranju generativnih nasprotniških mrež (GAN) na CT slikah**. Glavni cilj je bil razširiti obseg učnih podatkov za klasifikacijo pljučnih nodulov tako, da sva generirala nove slike s pomočjo dveh pristopov:
1. **DCGAN (skripta `gan.py`)**  
   - V tej implementaciji sva uporabila *Deep Convolutional GAN*, ki s preprosto arhitekturo generatorja in diskriminatorja generira 64×64 slike iz naključnega šuma.
   - Nova sintetična množica slik nato služi kot **augmentacija** pri učenju konvolucijske mreže za klasifikacijo.
2. **VQGAN (skripta `vqgan.py`)**  
   - Drugi pristop temelji na *Vector-Quantized GAN*, kjer sliko najprej zakodirava v **diskretno latentno** predstavitev, nato dekodirava nazaj v izhodno dimenzijo.
   - S tem pristopom sva želela preveriti, ali lahko **diskretizacija latentnega prostora** izboljša kakovost generiranih CT slik.

Po zaključku učenja sva pri vsaki metodi shranila vzorčne slike, ki so se generirale po posameznih epohah učenja. Spodaj so primeri generiranih slik iz `gan.py` in `vqgan.py`:

| Primer DCGAN | Primer VQGAN |
|--------------|--------------|
| ![DCGAN-slike](https://github.com/PainOfExistance/HouseAI/blob/main/assets/vqgan_epoch_50.png) | ![VQGAN-slike](https://github.com/PainOfExistance/HouseAI/blob/main/assets/epoch_80.png) |

Matej Habjanič: Izdelal sem napreden model za klasifikacijo raka na CT posnetkih z uporabo globokega učenja. Dodal sem predobdelavo slik, ki temelji na medicinskem znanju in potrebah (CLAHE, gamma korekcija). Uporabil sem moderno arhitekturo EfficientNetV2 z optimiziranim učenjem in ročno nastavljenimi utežmi za neuravnotežene razrede. V model sem vključil podrobno evalvacijo (ROC krivulje, Grad-CAM) za boljšo interpretabilnost.

Rene Rajzman: V sklopu tega šprinta sem začel z implementacijo ansambelske metode boosting z bootstrap vzorčenjem. Osnovni klasifikator je pristop, ki ga je razvil kolega Habjanič. Naletel sem na manjše težave zato še o rezultatih ne moram poročati

## 3. šprint
Matej Habjanič: Zaradi kolokvijev količina dela v tem tednu zmanjšana, so pa bili narejeni popravki kode in izboljšana berljivost kot tudi izboljšano evaluviranje modela s kreacijo pomočniških funkcij za posamezne metrike. Izboljšan tudi način kreiranja modelov za ansambelske metode katere bo kasneje Rene uredil in implementiral ter smiselno povezal in zapakiral.

Rene Rajzman: Zaradi kolokvijev in drugih obveznosti ta teden nisem opravil veliko dela. Raziskoval sem kako bi lahko odpravil težave slabih rezultatov Boosting ansambela ampak do implementacije rešitve še ni prišlo. Delo bom nadoknadil v naslednjem šprintu

Liam Mesarec, Kristina Čović: 
Nadaljevali sva razvoj generativnih modelov **DCGAN** in **VQGAN** za segmentacijo in razvrščanje CT-slik pljuč. Cilj je bil:
- Dodati več metrik za oceno kakovosti generiranih slik.
- Prilagoditi model, da generira slike **po posameznih razredih**.
- Preizkusiti kombinacije podatkovnih zbirk in različne hiperparametre.

**Priprava podatkov**
Kombinirala sva 2 dataseta : **The IQ-OTH/NCCD lung cancer dataset** + prej uporabljeni dataset.
- Dodala sva Normal in Benign slike iz novega dataseta v train/normal originalnega, da dobimo več raznovrstnih “zdravih” primerov  

**Implementacija GAN po razredih**
- Za vsak razred (`normal`, `adenocarcinoma`, …) se izvede ločen trening GAN-a.


**Primeri generiranih slik**
| Primer normal:| Primer squamous cell carcinoma: |
|--------------|--------------|
| ![](https://github.com/PainOfExistance/HouseAI/blob/main/assets/recon_ep046.png) | ![](https://github.com/PainOfExistance/HouseAI/blob/main/assets/recon_ep033.png) |


| Primer large cell carcinoma:| Primer adenocarcinoma |
|--------------|--------------|
| ![](https://github.com/PainOfExistance/HouseAI/blob/main/assets/recon_ep042.png) | ![](https://github.com/PainOfExistance/HouseAI/blob/main/assets/recon_ep047.png) |


## 4. Šprint

Rene Rajzman:
V tem šprintu sem razvil in preizkusil dva pristopa za izboljšanje klasifikacije, boosting ansambel in hibridni model.
Ponovno sem se lotil razvoja boosting ansambelskega pristopa na osnovi izjemno uspešnega EfficentNet modela z namenom dodatnega izboljšanja klasifikacije.Modeli so bili trenirani, shranjeni in povezani v ansambel. Kljub višji kompleksnosti in specializaciji posameznih modelov na napake predhodnikov se je rezultat napram samostojnemu modelu žal poslabšal. 
Implementiran hibridni pristop združuje napovedi treh modelov (EfficientNet, ResNet50 in MobileNetV3). Po implementaciji pristopa učenja ResNet50 in MobileNetV3 (katerih učenje je izvedel kolega Habjanič, saj je moj računalnik naletel na določene težave) sem vse tri obstoječe modele povezal v hibrid čigar testiranje na žalost ni pokazalo izboljšave napram EfficentNet modela.
