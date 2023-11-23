# HPylori-detector
Aquest projecte aborda la deteccio de la bacteria Helicobacter pylori, aquesta bacteria es la causa principal de gastritis i pot conduir a altres malalties greus, com úlcera gàstrica i fins i tot càncer. La principal forma de poder fer la seva detecció es a traves de l'anàlisi d'imatges histològiques amb tinció immunohistoquímica,un procés en el qual determinats anticossos tenyits s'uneixen a antígens de l'element
biològic d'interès. Aquesta anàlisi és una tasca que requereix molt temps per els professionals de la salut, es per aixó que farem us d'una xarxa neuronal on aplicarem l'us d'un ``AutoEncoder`` per a la detecció de patrons de tinció anòmals sencers tenyits inmunohistoquímicamente en imatges histològiques i fer la seva posterior classificació respecte si conte la bacteria o per el contarari esta sà.

## Codi
L'estructura del projecte es la següent:
1. ``main.py``: Conté el codi principal del projecte, a l'executarlo es posa en funcionament tot el sistema per entrenar/testejar el model d'autoencoder.
2. ``config.json``: Conte la configuració utilitzada durant el projecte.
3. ``handlers``:
   - ``__init__.py``: Imports llibreries.
   - ``cofiguration.py``: Carrega configuració per els paramatres del model i permet multiexecució.
   - ``data.py``: Crida per recuperar les dades del dataset a traves de les classe Dataset i crear els DataLoaders.
   - ``generator.py``: Genera objecte model i les seves funcions derivades per guardar-lo i carregar-lo.
   - ``train.py``: Entrenament del model.
   - ``test.py``: Test i metriques model.
4. ``objects``:
   - ``__init__.py``: Imports llibreries.
   - ``dataset.py``: Defineix classes dataset per carregar i guardar les dades.
   - ``model.py``: Defineix l'arquitectura del model.
5. ``models``: Contenidor per guardar els models generats en format .pickle.
6. ``plots``: Contenidor per guardar les figures referents a les metriques del model.


# Dataset

La base de dades QuirionDataBase amb la que s’ha treballat conte les carpetes ``CroppedPatches`` i ``AnnotedPatches`` amb els seus csv corresponents ``metadata.csv`` i ``window_metadata.csv``. 

La carpeta ``CroppedPatches`` conte una crapeta per cada pacient amb la identifiació B22-X on X es la id del pacient, la carpeta per cada pacient conte les seves imatges histològiques retallades. El csv corresponent a CroppedPatches conte la densitat de bacteria que trobem en cada pacient (*BAIXA,ALTA,NEGATIVA*) representat amb la seva id. Per altres banda tenim la carpeta ``AnnotedPtaches`` la qual es un subconjunt de CroppedPatches, la diferencia en aquest cas es que conte un csv amb la la seguent *informació id_pacient, numero_patch, positiu/negatiu* d'aquesta forma tenim un ground truth d'imatges etiquetades.

# Metedologia

La metedologia a seguir per classificar les imatges histològiques dels pacients i poder determinar la densitat de la bacteria que tenen consistira en fer l'us d'un autoencoder per la reconstrucció d'imatges. L'objectiu radera daquest autoencoder es overfitejar la reconstrucció d'imatges amb pacients amb densitat de bacteria negativa, la qual cose es tradueix en que les seves imatges histològiques no contentat Helicobacter. La presencia de Helicobacter en imatges es veu representada amb punts en el canal vermell de la imatge. Al overfitejar imatges de pacient que no contenen punts en aquest canal vermell la reconstrucció tampoc en tindra, de forma que al reconstruir imatges que si que continguin el bacteri la seva reconstrucció sera erronea ja que tampoc contindra els punts en el canal vermell de sortida.

Es podra saber quines son les imatges infectades mirant la frequencia dels seus punts en el canal vermell, es a dir, ``Fred = input (punts en el canal vermell) / outuput /(punts en el canal vermell)``. Si ``Fred > 1`` siginificara que tenim menys punts vermells en el canal de sortida que en el d'entrada, per tant la reconstrucció estara mal feta i sabrem que aquella imatge conte Helicobacter.

Finalment nomes caldra classificar el pacient segons la densitat de bacteria Helicobacter que contingui.

Es pot veure el procces mostrar en la *Figura 1*

#### Figura 1
![image](https://github.com/rauldaal/HPylori-detector/assets/61145059/09aa29f7-c41f-42ed-a5bd-04e1c46897d2)
*Pau Cano,Alvaro Caravaca,Debora Gil,and Eva Musulen.Schema of the main steps in the detection of H. pylori.
 https://arxiv.org/pdf/2309.16053.pdf*

## Dataloader 

Per tant pel desenvolupament del projecte s'ha definit una classe pare ``QuirionDataset`` amb dos classes filles ``CroppedDataset`` i ``AnnotedDataset`` les quals agafaran els datasets mencionats. I es dividiran de forma que la classe ``CroppedDataset`` nomes s's'utilitzara per l'entrenament del model i utilitzant nomes aquelles imatges de pacient amb una densitat de Helicobacter *negativa* i que no estiguin contingudes en el subconjunt ``d'AnnotedDataset``. Del total d'aquestes imatges negatives un 80% sutilitzaran per entrenament i el 20% restant per validació.

Per altre banda la classe ``AnnotedDataset`` agafara totes les seves imatges tant negatives com positives i s'utilitzaran per fer la seva classificació amb el model previament entrenat.

## Arquitectura model

L'arquitectura d'autoencoder que s'hautilitzar per l'entrenament del model es la seguent.

S'ha definit un codificador que pren una imatge d'entrada de 32x32 píxels i 3 canals (RGB). Aquesta imatge passa per dues capes *convolucionals* de 64 i 32 filtres i dues capes de *(max pooling)* per reduir la seva dimensió de 64x46 -> 32x32  -> 8x8 .I entre mig capes d'activació Relu, això permet capturar les característiques més importants de la imatge. Per tant finalment acabem en una representació de 8x8.

Respecte la definició del descodificador, pren aquesta representació de 8x8 i l'amplia utilitzant dues capes de convolució transposada de 32 i 64 filtres per aconseguir una imatge reconstruïda de 32x32 píxels. I entre mig una capa d'activació Relu. Finalment, una funció d'activació sigmoide s'aplica per assegurar que els valors de sortida estiguin entre 0 i 1.

## Determinació threshold 



## Classificació Imatges

### Metrqiues i resultats

## Classificació Pacient

### Metrqiues i resultats



Introducimos solo muestras negativas en el Autoencoder

Introducimos muestras positivas en el autoencoder

El error aumenta en X situación


