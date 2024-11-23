## Fork of https://github.com/NVlabs/sionna
Used for exfiltrating candidate path and choosing initial candidate paths to evaluate, for the ultimate goal of predicting a good starting array angle budget.

For our purposes the installation process should look like this (assuming cuda and video drivers are already configured)

Create conda env with python 3.10 or 3.11
```
conda activate [new env]
make install
pip install tensorflow[and-cuda]
pip install ipykernel
```