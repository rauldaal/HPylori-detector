# HPylori-detector
Aquest projecte aborda la deteccio de la bacteria Helicobacter pylori, aquesta bacteria es la causa principal de gastritis i pot conduir a altres malalties greus, com úlcera gàstrica i fins i tot càncer. La principal forma de poder fer la seva detecció es a traves de l'anàlisi d'imatges histològiques amb tinció immunohistoquímica,un procés en el qual determinats anticossos tenyits s'uneixen a antígens de l'element
biològic d'interès. Aquesta anàlisi és una tasca que requereix molt temps per els professionals de la salut, es per aixó que farem us d'una xarxa neuronal on aplicarem l'us d'un ``AutoEncoder`` per a la detecció de patrons de tinció anòmals sencers tenyits inmunohistoquímicamente en imatges histològiques i fer la seva posterior classificació respecte si conte la bacteria o per el contarari esta sà.

#### Figura 1
![image](https://github.com/rauldaal/HPylori-detector/assets/61145059/09aa29f7-c41f-42ed-a5bd-04e1c46897d2)
*Pau Cano,Alvaro Caravaca,Debora Gil,and Eva Musulen.Schema of the main steps in the detection of H. pylori.
 https://arxiv.org/pdf/2309.16053.pdf*

## Codi
El projecte conté els següents arxius *.py* i *.sav*:
1. ``matriculas_v3_OCR.py i matriculas_v3_svm.py``: Conté el codi principal del projecte, a l'executarlo es posa en funcionament tot el sistema de reconeixement automàtic de matrícules, ja sigui amb OCR o svm.
2. ``models.py``: Conte les funcions necessàries per crear els models de SVM de lletres i dígits.
3. ``DatasetMatriculaEspanyola.py``: Segmenta la fotografia que conté els caràcters amb la font de la matrícula espanyola, i les guarda en una carpeta per poder crear els models posteriorment.
4. ``lletresv4(7).sav``: Model SVM per les lletres


# Pasos

Introducimos solo muestras negativas en el Autoencoder

Introducimos muestras positivas en el autoencoder

El error aumenta en X situación

# Project structure

# Results
