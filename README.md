## T-DEV-810-PAR_26

M1 project at Epitech. 

# Data 

ChestXray data provided by the school. 
They are Kaggle data. 
These images (.jpeg) are thorax scans of healthy and sick patients (suffering from viral or bacterial pneumonia).


# Install requirement 

```
pip install . &&\
pip install -r requirements.txt
```

# Path requirement for jupyter issue

```
export PYTHONPATH="${PYTHONPATH}:/Users/lenaoudjman/Desktop/ZOIDBERG2.0/”
export PYTHONPATH="${PYTHONPATH}:/Users/mojgan/code/EPITECH/ZOIDBERG2.0/”
```

# Steps to follow 

- Creating an environment 
- Create a config.py to generalize paths ** NEVER PUSH!
- What is the model evaluated against? 
    - Accuracy 
    - F1 score ( mix between Recall and Precision )
    - AUC 
  => No overfitting / underfitting

- Pretest data 
    - Meme shape (150,150)
    - Color 
    - Batch : 32 ( Caution not for test)


# Styles 

On each python script:

```black path_file.py ```

```ruff check path_file.py ```
