
# Sistem za prepoznavo rakavih nodulov na pljučah z uporabo konvolucijskih nevronskih mrež nad podatki augmentiranimi z generativno nasprotniško mrežo

## Definicija problema
Pljučni rak je vodilni vzrok smrti zaradi raka. Pogosto je zaznan prepozno, ko so možnosti zdravljenja zelo omejene. Delimo ga na dva tipa, drobnocelični pljučni rak (angl. small cell lung cancer - SCLC) in nedrobnocelični pljučni rak (angl. non-small cell lung cancer - NSCLC) [[1]](#1). Najpogostejši in najboljši način za odkritje abnormalnosti v pljučih je nizkodozno računalniško tomografiranje (CT) [[2, 3]](#2). CT slike pregleda radiolog in na podlagi nadaljnih raziskav na pacientu ugotovi če ima rakave celice [[3]](#3). Zaradi velike možnosti človeške napake se je z razvojem nevronskih mrež začelo razvijati sisteme, ki s pomočjo umetne inteligence prepoznajo abnormalnosti na CT slikah [[4]](#4). Problem pri nevronskih mrežah je, da potrebujejo zelo veliko število označenih učnih podatkov, kar pa je pri medicinskih slikah pogosto težava. Za odpravitev te težave se v zadnjih letih na področju pljučnega raka eksperimentira z uporabo generativnih nasprotniških mrež (GAN) [[5, 6]](#5). GAN omogočajo učenje globokih reprezentacij brez obsežno označenih učnih podatkov. To dosežejo z izpeljavo signalov za povratno širitev napake (backpropagation) s pomočjo tekmovalnega procesa med dvema agentoma [[7]](#7).

- **ključne besede**: računalniško podprta diagnostika (CAD), konvolucijska nevronska mreža (CNN), globoko učenje, obdelava medicinskih slik, generativna nasprotniška mreža (GAN)
- **keywords**: Computer-aided diagnosis (CAD), Convolutional neural network (CNN), Deep learning, Medical image processing, Generative adversarial network (GAN)

## Pregled sorodnih del
Diagnoza pljučnega raka s pomočjo globokega učenja ali CNN temelji na klasificiranju škodljivosti pljučnih nodulov [[8]](#8). Za to potrebujemo zelo veliko množico učnih podatkov. Najpogosteje uporabljene v najboljših modelih v letih 2021 do 2024 so LC25000, LIDC-IDRI in LungCT-Diagnosis [[4]](#4). Veliko je tudi manjših množic kot so QIN LUNG CT in LUNA2016 [[9, 10]](#9). Ker so učne množice majhne s samo nekaj tisoč slikami pljučnih nodulov, so podatki pogosto augmentirani s transformacijami nad slikami [[11]](#11). Novejša metoda augmentacije je GAN, ki namesto geometrijskega transformiranja tvori nove slike s tekmovanjem med dvema nevronskima mrežama kjer ena mreža generira podatke, druga pa poskuša ugibati če so ti podatki prišli iz originalne učne množice ali so bili generirani [[11, 12]](#11). StyleGAN generira nove slike na podlagi značilk učnih podatkov, kar bi lahko bilo uporabljeno za učenje značilk pljučnih nodulov [[13]](#13). CycleGAN lahko generira slike, ki so bile zajete v različnih domenah, kot je pretvorba MRI v CT [[14]](#14). Za preverjanje kako realistična je slika se uporablja Fretchet Inception Distance [[13, 14]](#13). Da se izognemo problemu male množice učnih podatkov lahko uporabimo tudi prej trenirane nevronske mreže. Tudi te lahko dosegajo natančnost nad 0.95 vendar zahtevajo dobro optimizacijo z uporabami metod prenosa znanja [[15]](#15). V eni študiji je VGG-16 dosegel natančnost 0.981, MobileNet  0.945, ResNet50 0.925 in InceptionV3 0.92 [[16]](#16). Večjo natančnost so zasledili pri modelih CNN GD z 0.9786 [[17]](#17) in 0.9945 s CNN [[18]](#18). Zelo dobro natančnost ima tudi LDNNET z natančnostjo 0.988396 [[19]](#19). U-Net modeli so popularni za medicinske slike vendar imajo manjšo natančnost med 0.8 do 0.9 [[20]](#20). 

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
    3. razvojna iteracija
        - testiranje z drugimi CNN pred-trenirani modeli
        - optimizacija
    4. razvojna iteracija
        - uporabniški vmesnik za aplikacijo
        - primerjava rezultatov z drugimi modeli
        - pisanje članka
- **UML diagram rešitve:**

## ![houseai](https://github.com/user-attachments/assets/2b8e0595-ee9a-4c0d-9393-afec296d37c2)



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

