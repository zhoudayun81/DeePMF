# DeePMF
Our code uses Python3, to ensure smooth running, please make sure your python environment is at least Python 3.9.19.

We also used Optuna (https://optuna.org/). Other package information is available in our paper.

## Running experiment
A quick cheating command to install all packages using pip to run our code is: 
<pre>
  <code id="install-command">pip install numpy=1.26.4 pandas=2.2.2 matplotlib scikit-learn==1.5.1 scipy=1.13.1 optuna pm4py</code>
</pre>
<button onclick="copyToClipboard('#install-command')"></button>
This should allow you to install all the required packages at once.

Please make sure you keep the folder structures and go to [code folder](https://github.com/zhoudayun81/DeePMF/tree/main/upload/code) (```.\upload\code\```), then run: 
<pre>
  <code id="install-command">python3 main.py</code>
</pre>
<button onclick="copyToClipboard('#install-command')"></button> to initiate the experiment run.

You can modify the settings to different parameters for other experiment, and we left the [configuration file](https://github.com/zhoudayun81/DeePMF/tree/main/upload/code/config.ini) (```.\upload\code\config.ini```) as the default settings that we used and described in our paper. Explanations are provided in the file for each parameter that may not be intuitive to understand.
<hr />

Please note that our licence is under GNU Affero General Public License v3.0. 
If you need assistance in running the code or you are interested in the experiment or further collaboration, feel free to contact [Wenjun](mailto:wenjun.zhou@unimelb.edu.au?subject=[DeePMF]).
