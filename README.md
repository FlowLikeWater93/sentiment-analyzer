# sentiment-analyzer
<br>
This project was conducted as part of udacity's Data scientist nanodegree
<br><br>
Please download the dataset from <a href="https://www.yelp.com/dataset" target="blank">yelp</a> before running the project source code. Place the file called "yelp_academic_dataset_review.json" only in the yelp_dataset directory. The data has not been included in this repository due to its large size (~5GB)
<br><br>
Building a reputable and popular brand is not an easy job. In order to do so, businesses must constantly understand and gauge the feedback of their audience. Businesses and especially large brands tend to have an abundance of text data from their accounts on social media platforms. To make the most out of the data, we can utilize pretrained NLP models to build sentiment profiles from text data gathered online.
<br><br>

### Goal of this project:
Build a sentiment analysis model that using LSTMs that can label text input as positive or negative

## Running the project files
1- analysis.py :
<br>reads and analyzes the dataset yelp_academic_dataset_review.json. can save graphs in graphs directory
<br><br>
Make sure you downloaded the dataset from the link above and placed the json file in the yelp_dataset directory
<br><br>
Navigate to the root directory
<br><br>
```
python analysis.py
```
<br>
2- model.py :
<br>This file trains the model on the dataset yelp_academic_dataset_review.json and saves the model in the LSTM.pth file
<br><br>
Navigate to the "model" directory and run the following command
<br><br>

```
python model.py
```
<br><br>

## Installations
In order to clone the project files and run them on your machine, a number of packages and libraries must be installed
<br><br>
**1- python 3.10**
<br><br>
**2- Pandas**
<br>
  To install
<br>
```
# conda
conda install -c conda-forge pandas
# or PyPI
pip install pandas
```
<br>

**3- Numpy**
<br>
  To install
<br>
```
# conda
conda install -c anaconda numpy
# or PyPI
pip install numpy
```
<br>

**4- scikit-learn**
<br>
  To install
<br>
```
# conda
conda create -n sklearn-env -c conda-forge scikit-learn
# or PyPI
pip install -U scikit-learn
```
<br>

**5- matplotlib**
<br>
  To install
<br>
```
# conda
conda install -c conda-forge matplotlib
# or PyPI
pip install -U matplotlib
```
<br>

**6- torch**
<br>
  2.1.1+cpu - To install
<br>
```
# conda
conda install pytorch torchvision torchaudio cpuonly -c pytorch
# or PyPI
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
<br>

**7- torchtext**
<br>
  0.16.1+cpu - To install
<br>
```
# conda
conda install -c pytorch torchtext
# or PyPI
pip install torchtext
```
<br>

## Project files
This repository contains the following :
<br><br>
**analysis.py**
<br>
reads and analyzes the dataset yelp_academic_dataset_review.json
can save graphs in graphs directory
<br>
**helperFunctions.py**
<br>
Contains helper function for analysis.py
<br>
**model/LSTM.py**
<br>
LSTM class. specifies model hyper parameters
<br>
**model/model.py**
<br>
Run the file to train the model on the dataset yelp_academic_dataset_review.json
<br>
**model/vocab.pth**
<br>
This file contains the vocabulary of our whole corpus used in training
<br>
**model/LSTM.pth**
<br>
This file contains final model parameters to be loaded later for production or further training
<br>
**model/epoch_metrics.pth**
<br>
This file contains metrics for each epoch of training stored in lists
<br>
**graphs directory**
<br>
you will find the graphs saved from the analysis phase here
<br>
**yelp_dataset directory**
<br>
contains the dataset as well as the user agreement


## Acknowledgements
<a href="https://pytorch.org/get-started/locally/">torch download</a>
<br>
