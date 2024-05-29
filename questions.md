# Preguntes del projecte


## Dia 8/5/2024
1. Anàlisi dels dos starting points:
   1. Què fa cada un
   2. Diferències
3. Quines millores apliquen cada starting point en relació als models preentrenats?
4. Quines millores podríem fer per obtenir millors resultats?

### Feedback:
Mètrica d'avaluació: No hi ha mètrica perquè és subjectiva. Valoracions subjectives justificades. 
Loss: què fa, què minimitza, entendre-ho bé. Què està avaluant exactament. 
   És la unica manera que tinc d'avaluar? Hi han altres?
   Idea més enllà per poder dir si dues fotos tenen el mateix estil: part de l'embedding et diu l'estil. Treure l'estil i predir l'estil. 
Embeddings: què son, què són les xarxes. Començar amb el model AdaIn. 
Veure curves de loss. 
El pes decideix si es guanya o perd estil.
Imatge neta + imatge bruta : la loss compara la imatge bruta menys la neta i minimitza la diferència. Pero realment, què és el que vol que sigui petit? Com controlo que es barregin bé. 

Conclusió: entendre el model i tornar la setmana que ve amb preguntes noves més consistents. 

## Dia 15/5/2024
Nou enfoc
1. Detecció de les layers més influents en la detecció d'estil
2. Imatge objectiu: soroll / imatge blanca / color
3. Finetuning i feature extraction.  

Agafar model inception al encode decode i despres amb la vgg mirem la similitud entre la imatge de contingut, d'estil i generada. 

Pagina web "papers with code"


## 22/5/2024