# DeePMF
Our code uses Python3, to ensure smooth running, please make sure your python environment is at least Python 3.9.19.
We also used Optuna (https://optuna.org/). 
Other package information is available in our paper. 
A quick cheating command to install all packages using pip to run our code is:
```pip install numpy=1.26.4 pandas=2.2.2 matplotlib scikit-learn==1.5.1 scipy=1.13.1 optuna pm4py```
This should allow you to install all the required packages at once.
Please make sure you keep the folder structures and go to ```cd .\upload\code\```, then run ```python3 main.py``` to initiate the experiment run.
You can modify the settings to different parameters for other experiment, and we left the ```.\upload\code\config.ini``` file as the default settings that we used and described in our paper.
