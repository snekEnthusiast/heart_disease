## requirements
python 3.10.12 \
ucimlrepo \
tensorflow \
pandas \
pathlib
>pip install ucimlrepo tensorflow pandas pathlib
## setup
run data.py before createmodel to download data \
or simply use the pipeline tool \
data with missing (NaN) values will be ignored
## other
training/test division is 8/2 \
*models* folder contains pretrained models, with names representing their auc metric \
*data.py* contains a description of features
## citation
Janosi,Andras, Steinbrunn,William, Pfisterer,Matthias, and Detrano,Robert. (1988). Heart Disease. UCI Machine Learning Repository.
https://doi.org/10.24432/C52P4X
## flow
this repo is also equipped with a pipeline tool, although it really isn't necesarry
>pip install metaflow

>python3 flow.py run