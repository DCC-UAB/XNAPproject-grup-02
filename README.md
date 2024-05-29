[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/L30CyvB9)
# Neural Style transfer (NST)

Al final fem el readme en un document a part que s'ens fa més facil manipular.

## Introducció

En les darreres dècades, els avenços en l'aprenentatge profund (deep learning) han revolucionat el camp del processament d'imatges, obrint noves possibilitats creatives i tecnològiques. Una de les aplicacions més fascinants i creativa és la transferència d'estil, un procés pel qual el contingut d'una imatge es fusiona amb l'estil d'una altra, generant resultats visualment sorprenents. 

Aquest treball explora i analitza un metode en concret amb l'objectiu de proporcionar una comprensió profunda dels mètodes existents, les seves aplicacions, i els reptes associats. Es començarà amb una revisió exhaustiva dels fonaments teòrics i del algoritma de la transferència d'estil.Finalment, es presentarà un estudi de cas pràctic en el qual s'aplicaran aquestes tècniques a un conjunt d'imatges específiques, analitzant els resultats obtinguts i discutint les possibles aplicacions futures d'aquesta tecnologia.

----

Aquest projecte és un projecte de Transferència d'estils en imatges de deep learning. Té com a objectiu principal transferir estils en imatges de vista de carrer, combinant característiques textuals i visuals.

El que fa aquesta transferència d'estils és que pren una imatge de referència (anomenada "estil") i una altra imatge (anomenada "contingut"), i combina els aspectes visuals de l'estil amb el contingut de la imatge de manera que el resultat sembli que ha estat pintat amb l'estil de la imatge de referència. L'objectiu és aconseguir el millor resultat possible.

El projecte utilitza un conjunt de dades de 42.129 imatges proporcionades per 195 artistes, que es pot trobar al següent enllaç: https://paperswithcode.com/dataset/wikiart. Això proporciona una gran varietat d'estils visuals per entrenar el model.

El starting-point on ens vasem per realitzar l'implementació esta definit en el github ENLAZEEE

Aquests enllaços et portaran als repositoris de GitHub relacionats amb el projecte. Potser hi trobaràs codi, documentació i altres recursos que t'ajudaran a entendre millor com es duu a terme la transferència d'estils en aquest context específic.

## Objectius



## Com funciona una NST?

### Introducció

La transferència d'estil es va popularitzar amb els treballs pioners de Gatys et al. (2015), que van ser el primer en demostrar com una xarxa neuronal convolucional (CNN) podia descompondre i recombinar les característiques de contingut i estil de les imatges. Les CNN són el tipus de de xarxes neuronals més potent per a la classificació i l'anàlisi d'imatges. Les seves aplicacions han superat molts límits i s'ha demostrat que són l'element crític de moltes aplicacions habilitades per a l'aprenentatge profund d'avui en dia. A un nivell molt alt, les CNN poden aprendre les representacions a nivell de característiques internes de les imatges les quals s'alimenten. Són molt potents, i no només funcionen per a classificar imatges, sinó que també funcionen per la construcció d'imatges.

Aquesta tècnica inicial, tot i ser potent, més endevant es van millorar els resultats amb altres xarxes. Des de llavors, s'han desenvolupat nombroses variacions i millores, incloent-hi les xarxes generatives per a la transferència en temps real, els models adaptatius multiestil i les xarxes adversaries generatives (GANs), entre d'altres.

Tot i aixó, 


L'objectiu d'aquest treball és proporcionar una visió integral i pràctica de la transferència d'estil en l'aprenentatge profund, contribuint al desenvolupament i a la comprensió d'aquest camp emergent i oferint eines útils per a la seva aplicació en diversos contextos creatius i tecnològics.


Aquest projecte es tracta de generar imatges a partir d'una imatge amb contingut i una imatge amb un estil concret. Volem combinar-les de manera que el model entrenat generi una imatge que contingui el contingut de la imatge amb contingut i l'estil de la imatge de l'estil. Per tant, hem d'aconseguir un model amb el contingut i l'estil correctes. 

![Imatge1](img/Img1.png)

**Figura 1**: Imatge esquerre - imatge contingut, imatge dreta - imatge estil.

Per fer el procés de transferència de l'estil de la imatge estil a la imatge contingut la CNN s'entrena per optimitzar una funció de pèrdua (loss). Quan les imatges es "fusionen", un exemple d'output podria ser: 

![Imatge2](img/Img2.png)

Això ho podrem aconseguir, com ja hem dit, a partir de CNN, Xarxes Neuronals Convolucionals. A mesura que més profunda és la capa que s'entrena i més filtres té, més característiques captura. Aquest procés també es pot inicialitzar amb l'ús d'una imatge amb soroll com a imatge inicial en la transferència d'estil neural per iniciar el procés de manera neutra, evitar òptims locals i donar flexibilitat als resultats finals. A través de l'optimització iterativa, aquesta imatge amb soroll es transforma progressivament per reflectir millor tant el contingut com l'estil de les imatges de referència.

És una eina molt útil per plasmar estils artístics determinats en qualsevol imatge. 

### Aprofunditzant

Una CNN és un conjunt de capes convolucionals i capes de pooling. Les capes convolucionals extreuen característiques altament complexes d'una imatge donada, mentre que les capes de pooling (en les que es redueix dimensionalitat) descarten informació espacial i detalls irellevants. L'efecte d'això és que ajuda la CNN a aprendre el contingut d'una imatge per sobre de qualsevol cosa específica com el color, la textura, etc. A mesura que anem aprofundim en les capes (layers), la complexitat de les característiques augmenta i les capes convolucionals més profundes sovint són les que s'entendrien com a **representacions del contingut**.

El concepte d'estil es pot concebre com les propietats d'una imatge (textura, el color...), per obtenir-ho es calculen les correlacions entre les capes convolucionals perquè ens diuen com de similars/relacionades són dues o més variables (característiques). Cada capa convolucional està formada per diversos mapes de característiques, i per cada un es mesura com de fortes estan relacionades aquestes característiques amb les altres de la mateixa capa. A partir d'aquestes relacions i magnituds es pot estimar sobre certs aspectes i relacions entre les imatges: formes comunes, mapping de colors... Els trets / similituds que es troben entre mapes de característiques composen l'estil de la imatge. Això ens permet obtenir una representació multi-escala de la imatge donada que es centra en característiques espacials com la textura,el color, i algunes formes i patrons, que és el que principalment defineix un estil.

## Com quantifiquem la Imatge de Contingut?


/*Completar i plantejar millor aquesta explicació*/

Una arquitectura que es podria utilitzar és una VGG-19 per extreure tant les característiques de contingut com d'estil de les imatges de contingut i d'estil respectivament. 

Per obtenir les característiques de contingut, s'utilitza la segona capa convolucional del quart bloc (de capes convolucionals). Amb aquestes característiques de contingut es compararan amb una imatge objectiu per mesurar la pèrdua de contingut. Pero... d'on treiem la imatge objectiu?

Aquesta imatge objectiu és important per combinar les característiques de contingut i d'estil en una única imatge. Aquesta imatge pot ser simplement un espai en blanc o una còpia de la imatge de contingut. 

### Mètriques: Càlcul la Pèrdua de Contingut

Com l'evaluació del resultat és subjectiva, no podem avaluar l'eficiència del model a partir d'una mesura. El que podem fer és buscar una alternativa, i per tant,  ens centrarem en com varia la Loss i en optimitzar de similar és la imatge de soroll random amb la imatge de contingut.

Per poder dur a terme tot l'esmentat anteriorment es necessita una funció de pèrdua personalitzada al model que s'optimitzarà per obtenir una imatge de sortida suau construïda a partir de les imatges de contingut i d'estil. 

Aquesta funció consisteix en:

- **Pèrdua de Contingut**, que assegura que la quantitat de contingut es preservi.
- **Pèrdua d'Estil**, que s'encarrega de la quantitat d'estil que es transfereix a la imatge objectiu.

La pèrdua de contingut es defineix de la manera següent:

- $T_c$ : la imatge objectiu
- $C_c$ a la imatge de contingut

Aquesta funció de pèrdua et dona una mesura de com de lluny estan les característiques de les imatges de contingut i objectiu una de l'altra. La xarxa neuronal intentarà minimitzar aquesta loss. 



## Code structure
You must create as many folders as you consider. You can use the proposed structure or replace it by the one in the base code that you use as starting point. Do not forget to add Markdown files as needed to explain well the code and how to use it.

## Example Code
The given code is a simple CNN example training on the MNIST dataset. It shows how to set up the [Weights & Biases](https://wandb.ai/site)  package to monitor how your network is learning, or not.

Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) file has all the required dependencies. Run the following command: ``conda env create --file environment.yml `` to create a conda environment with all the required dependencies and then activate it:
```
conda activate xnap-example
```

To run the example code:
```
python main.py
```

## Bibliografia

- - https://github.com/HalbertCH/IEContraAST
- https://github.com/elleryqueenhomels/arbitrary_style_transfer/blob/master/README.md
- https://towardsdatascience.com/how-do-neural-style-transfers-work-b76de101eb3
- https://www.datacamp.com/tutorial/implementing-neural-style-transfer-using-tensorflow?utm_source=google&utm_medium=paid_search&utm_campaignid=21057859163&utm_adgroupid=157296744657&utm_device=c&utm_keyword=&utm_matchtype=&utm_network=s&utm_adpostion=&utm_creative=698229405991&utm_targetid=dsa-2218886984100&utm_loc_interest_ms=&utm_loc_physical_ms=1005435&utm_content=&utm_campaign=230119_1-sea~dsa~tofu_2-b2c_3-es-lang-en_4-prc_5-na_6-na_7-le_8-pdsh-go_9-na_10-na_11-na-may24&gad_source=2&gclid=CjwKCAjwl4yyBhAgEiwADSEjeK1TcdnTtgpyEy8gIM71JAoq7XHXdxppv78CSWKpiTtpinE8EeYx0RoC8lkQAvD_BwE

## Contributors
Mireia Sempere, Maria Castellanos, Irene Castrillo

Xarxes Neuronals i Aprenentatge Profund
Grau d'Enginyeria de Dades, 
UAB, 2024

