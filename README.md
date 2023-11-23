# HPylori-detector
Aquest projecte aborda la deteccio de la bacteria Helicobacter pylori, aquesta bacteria es la causa principal de gastritis i pot conduir a altres malalties greus, com úlcera gàstrica i fins i tot càncer. La principal forma de poder fer la seva detecció es a traves de l'anàlisi d'imatges histològiques amb tinció immunohistoquímica,un procés en el qual determinats anticossos tenyits s'uneixen a antígens de l'element
biològic d'interès. Aquesta anàlisi és una tasca que requereix molt temps per els professionals de la salut, es per aixó que farem us d'una xarxa neuronal on aplicarem l'us d'un ``AutoEncoder`` per a la detecció de patrons de tinció anòmals sencers tenyits inmunohistoquímicamente en imatges histològiques i fer la seva posterior classificació respecte si conte la bacteria o per el contarari esta sà.

#### Figura 1
![image](https://github.com/rauldaal/HPylori-detector/assets/61145059/09aa29f7-c41f-42ed-a5bd-04e1c46897d2)
*Pau Cano,Alvaro Caravaca,Debora Gil,and Eva Musulen.Schema of the main steps in the detection of H. pylori.
 https://arxiv.org/pdf/2309.16053.pdf*

## Codi
L'estructura del projecte es la següent:
1. ``main.py``: Conté el codi principal del projecte, a l'executarlo es posa en funcionament tot el sistema per entrenar/testejar el model d'autoencoder.
2. ``config.json`: Conte la configuració utilitzada durant el projecte.
3. ``handlers``:
   3.1. ``__init__.py``: Imports llibreries.
   3.2. ``cofiguration.py``: Carrega configuració per els paramatres del model i permet multiexecució.
   3.3. ``data.py``: Crida per recuperar les dades del dataset a traves de les classe Dataset i crear els DataLoaders.
   3.4. ``generator.py``: Genera objecte model i les seves funcions derivades per guardar-lo i carregar-lo.
   3.5. ``train.py``: Entrenament del model.
   3.6. ``test.py``: Test i metriques model.
4. ``objects``:
   4.1. ``__init__.py``: Imports llibreries.
   4.2. ``dataset.py``: Defineix classes dataset per carregar i guardar les dades.
   4.3. ``model.py``: Defineix l'arquitectura del model.
5. ``models``: Contenidor per guardar els models generats en format .pickle.
6. ``plots``: Contenidor per guardar les figures referents a les metriques del model.


# Pasos

Introducimos solo muestras negativas en el Autoencoder

Introducimos muestras positivas en el autoencoder

El error aumenta en X situación

# Project structure

# Results
