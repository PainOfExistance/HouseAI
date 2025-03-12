
# HouseAI

## zapiski
- detekcija NSCLC in SCLC, pazimo da SCLC predstavlja 80% primerov
- https://link.springer.com/article/10.1007/s11831-024-10141-3
    - Generative adversarial network (GAN) za generiranje podatkov. TO MA DEJANSKO VEZE Z AGENTI. lahko uporabimo s CNN
    - pogosto uporabljeni CNN: VGG, alexnet, resnet

- podatki
    - LUNA16
    - lndb grand challenge
    - LC25000
    - LIDC-IDRI
    - NLST
    - QIN LUNG CT (diagnoze NSCLC)

- GAN
    - tensorflow: StyleGAN
    - če rabimo pretvoriti slike iz ene domene v drugo (npr. različen način zajema slik) potem imamo CycleGAN
    - Fréchet inception distance za preverjanje realizma slike

## Definicija problema
**Poglavje na vsebuje opis problema, kjer ga v nekaj stavkih definirate. Definirajte ali izbrana problematika izhaja iz realnega sveta oziroma je ta umetno ustvarjena. Problem skušajte tudi formulirati z matematično notacijo. Na koncu poglavja podajte tudi pet ključnih besed, ki se nanašajo na vašo rešitev, problem in metodo evaluacije vaših rešitev problema. Pri pripravi ključnih besed si pomagajte tudi s pregledom sorodnih del.**

Pljučni rak je vodilni vzrok smrti zaradi raka. Pogosto je zaznan prepozno, ko so možnosti zdravljenja zelo omejene. Delimo ga na dva tipa, drobnocelični pljučni rak (angl. small cell lung cancer - SCLC) in nedrobnocelični pljučni rak (angl. non-small cell lung cancer - NSCLC) [[1]](#1). Najpogostejši in najboljši način za odkritje abnormalnosti v pljučih je računalniško tomografiranje (CT). CT slike pregleda radiolog in na podlagi nadaljnih raziskav na pacientu ugotovi če ima rakave celice [[2]](#2). Zaradi velike možnosti človeške napake se je z razvojem nevronskih mrež začelo razvijati sisteme, ki s pomočjo umetne inteligence prepoznajo abnormalnosti na CT slikah [[3]](#3). Problem pri nevronskih mrežah je, da potrebujejo zelo veliko število označenih učnih podatkov, kar pa je pri medicinskih slikah pogosto težava. Za odpravitev te težave se v zadnjih letih na področju pljučnega raka eksperimentira z uporabo generativnih nasprotniških mrež (GAN) [[4, 5]](#4). GAN omogočajo učenje globokih reprezentacij brez obsežno označenih učnih podatkov. To dosežejo z izpeljavo signalov za povratno širitev napake (backpropagation) s pomočjo tekmovalnega procesa med dvema agentoma [[6]](#6).

- **ključne besede**: računalniško podprta diagnoza, konvolucijska nevronska mreža (CNN), globoko učenje, obdelava medicinskih slik, generativna nasprotniška mreža (GAN)
- **keywords**: Computer-aided diagnosis, Convolutional neural network (CNN), Deep learning, Medical image processing, Generative adversarial network (GAN)

## Pregled sorodnih del
S pomočjo spletnih iskalnikov kot so Google Scholar ali iskalnikov založnikov znanstvenih revij. Rezultat pregleda sorodnih del je seznam ključne literature vaše problematike in rešitve problema. Pregled evaluacije rešitve problema se navezuje na najdeno literaturo. Rezultat tega dela naloge je **seznam metod evaluacije rešitve problema.**


## Načrt rešitve

- **skupina:** 1
- **sodelavci:** Matej Habjanič, Kristina Čović, Rene Rajzman, Liam Mesarec
- **repozitorij:** https://github.com/PainOfExistance/HouseAI
- **programski jezik:** Python
- **opravila**
    1. razvojna iteracija 1
    2. razvojna iteracija 2
    3. razvojna iteracija 3
    4. razvojna iteracija 4
- **opis rešitve:** UML diagram

## Viri
[1] “Lung cancer.” Accessed: Mar. 12, 2025. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/lung-cancer \
[2] R. Nooreldeen and H. Bach, “Current and Future Development in Lung Cancer Diagnosis,” International Journal of Molecular Sciences, vol. 22, no. 16, Art. no. 16, Jan. 2021, doi: 10.3390/ijms22168661. \
[3] S. L. Tan, G. Selvachandran, R. Paramesran, and W. Ding, “Lung Cancer Detection Systems Applied to Medical Images: A State-of-the-Art Survey,” Arch Computat Methods Eng, vol. 32, no. 1, pp. 343–380, Jan. 2025, doi: 10.1007/s11831-024-10141-3. \
[4] N. Ghaffar Nia, E. Kaplanoglu, and A. Nasab, “Evaluation of artificial intelligence techniques in disease diagnosis and prediction,” Discov Artif Intell, vol. 3, no. 1, p. 5, Jan. 2023, doi: 10.1007/s44163-023-00049-5. \
[5] Q. Jin, H. Cui, C. Sun, Z. Meng, and R. Su, “Free-form tumor synthesis in computed tomography images via richer generative adversarial network,” Knowledge-Based Systems, vol. 218, p. 106753, Apr. 2021, doi: 10.1016/j.knosys.2021.106753. \
[6] A. Creswell, T. White, V. Dumoulin, K. Arulkumaran, B. Sengupta, and A. A. Bharath, “Generative Adversarial Networks: An Overview,” IEEE Signal Processing Magazine, vol. 35, no. 1, pp. 53–65, Jan. 2018, doi: 10.1109/MSP.2017.2765202. \
