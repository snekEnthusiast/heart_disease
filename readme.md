## requirements
python 3.10.12 \
pandas \
numpy \
sklearn \
pathlib \
zipfile
>pip install pandas numpy sklearn pathlib zipfile
## setup
run data.py before createmodel.py to download data \
or simply use the pipeline tool
## other
training/test division is 8/2 \
*models* folder contains pre-trained models, with names representing their AUC metric \
*data.py* contains a description of features
## citation
Janosi,Andras, Steinbrunn,William, Pfisterer,Matthias, and Detrano,Robert. (1988). Heart Disease. UCI Machine Learning Repository.
https://doi.org/10.24432/C52P4X
## flow
this repo is also equipped with a pipeline tool, although it isn't necessary
>pip install metaflow

>python3 flow.py run