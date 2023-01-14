Detecção de Logotipos da Coca Cola ou Pepsi com Redes Neurais Convolucionais

Para trabalho final de machine learning foi feito um modelo com redes neurais convolucionais para reconhecimento de objetos dentro da imagem, as redes neurais convolucionais são naturalmente utilizadas quando trabalhamos com imagens, o motivo para isso é que elas utilizarem cálculos específicos entre os nós e acabam, normalmente, tendo uma precisão melhor que as outras.

As limitações dessa aplicação é a falta de um maior dataset, apesar de ter uma boa quantidade de imagens precisamos de muito mais que 1000 para um bom treino. Para ajudar e ampliar o dataset foi feito um pré-processamento onde aplicamos alguns filtros nas imagens e efetuamos rotações nelas para aumentar a precisão na detecção do logotipo. 

Para mostrar melhor a aplicação podemos rodar dois arquivos o train.py, para treinar o modelo e salvar ele para futuras utilizações, e depois podemos utilizar o connect.py que vai ativar a webcam do pc para a detecção da logo, ele funciona com a tecla ESC, toda vez que ela é clicada é tirado um frame da câmera,esse frame é analisado pelo modelo, essa analise retorna se a imagem mostrada tem mais chances de possuir a logo a coca ou da pepsi.

Caso você não tenha uma webcam pode utilizar do arquivo load_model.py onde ele dá um load no modelo criado, dessa forma você consegue prever todas as imagens que estão no niv.
Para melhorar o exemplo coloquei quatro imagens, duas da coca e duas da ps.

Lembrando que a precisão está longe de ser perfeita, se você segurar uma garrafa provavelmente ele vai dar um valor grande para os dois, é necessário para uma melhor precisão mostrar bem a logo.


PS: todos as lib necessárias estão no poetry só precisa executar o comando “peotry install”.
```zsh
poetry install
```

LOAD MODEL:
```zsh
python3 load_model.py
```

TRAIN MODEL:
```zsh
python3 detection_logo_type.py
```

Se você quiser utilizar a webcam para teste pode rodar o connect.py e apertar
ESC para capturar o frame da webcam.
```zsh
python3 connect.py

PRESS(ESC) -> capture the frame
```
