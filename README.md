# dsi-final-project

[![Generic badge](https://img.shields.io/badge/Python-3.6.7-<COLOR>.svg)](https://shields.io/)

## Descripción
El objetivo de nuestra práctica ha sido desarrollar un sistema de detección de bulos basados en vídeos simulados (_fake news_). Para ello, se han combinado varias técnicas de IA como son el reconocimiento facial, el reconocimiento del discurso y el reconocimiento de la voz. 

## Dependencias
Se ha mantenido un fichero de las librerías usadas denominado `requirements.txt`, en la raíz del proyecto. Para instalarlo podemos crearnos un entorno nuevo (recomendable) o ejecutar desde la terminal directamente. El comando sería el siguiente:

```shell
$> pip install -r requirements.txt
```

## Ejecución

1. Entrenamiento del módulo de reconocimiento facial, desde el directorio del módulo.
```shell
$> cd dsi_facial_recognition
$> python face-train.py
```

2. Entrenamiento del módulo de reconocimiento de voz, desde el directorio del módulo
```shell
$> cd dsi_speaker_recognition
$> python train_models.py
```	

3. Ejecución de la aplicación, desde la raíz.
```shell
$> python mainApp.py
```

## Autores
- Mario Sánchez García (NIA: 100315075)
- Juan Alonso Machuca González (NIA: 100317131)
