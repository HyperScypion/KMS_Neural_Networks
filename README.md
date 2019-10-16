# Polish:

## Krótko o repozytorium:
Repozytorium zostało założone dla wszystkich osób uczęszczających na warsztaty z uczenia maszynowego
prowadzone przez Koło Matematyki Stosowanej na Uniwersytecie Mikołaja Kopernika w Toruniu.
Znajdziecie tu między innymi kody z zajęć jak również i prezentacji, przydatne jupyter notebooki, a także
zadania do sprawdzenia swojej wiedzy. 

## Wstępny plan: 
- Wprowadzenie do KMS :ballot_box_with_check:
- Wstęp do Pythona czyli jak rozmawiać z wężem 1/2 :snake:
- Wstęp do Pythona czyli jak rozmawiać z wężem 2/2 :snake:

## Drzewo katalogów
```
KMS_Neural_Networks
│   LICENSE
│   README.md
│   requirements
│
└───codes
│   │
│   └───keras
│   │    │   irys.py
│   │
│   └───pure_python
│   │    │   activations.py
│   │    │
│   │    └───percetrons
│   │           │   perceptron.py
│   └───pytorch
│   │    │   irys.py
│   │
│   └───tensorjs
│   │    │   index.html
│   │    │    ...
│   │
└───lectures
│   │   first_lecture
│   │    │
│   │    │   links.txt
│   │    └───presentation
│   │           │   presentation.pptx
│   │   second_lecture
│   │    │
│   │    └───codes
│   │           │   ...
│   │    │
│   │    └───notebooks
│   │           │   ...
│   │    │
│   │    └───presentation
│   │           │   ...
└───notebooks
│   │   hello_from_jupyter.ipynb
│   │
│   └───pytorch
│   │    │
│   │    │   mnist_mlp.ipynb
│   │    └───MNIST
│   │           │   ...
```

## Instalacja modułów za pomocą pip-a:
### Za pomocą pliku requirements
```bash
$ pip3 install -r <file_name>
```
### Za pomocą nazwy modułu
```bash
$ pip3 install <module_name>
```
### Sposób alternatywny
```bash
$ python3 -m pip install <module_name>
$ python3 -m pip install -r <file_name>
```

### Instalacja za pomocą condy
```bash
$ conda install <module_name>
```

### Sposób alternatywny
```bash
$ conda install --file <file_name>
```
## Instalacja Jupyter-a
### Przez pip-a
Jupyter jest podany jako jeden z modułów w pliku requirements.
