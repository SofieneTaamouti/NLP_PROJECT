# NLP_PROJECT

We present the different files of this repository.

**The logistic classifiers**

The three logistic classifiers are trained in model_logistic_tfidf.py, model_logistic_stylo.py and model_logistic_both.py. We then test the performances of all three models on the hold-out sample (in Eval_logistic_on_holdout.ipynb) and the Medium dataset (Eval_logistic_on_Medium.ipynb). 

For stylometric features, we defined a function that computes  them in the stylometric_features.py script.

To avoid retraining the model everytime, we stored the output of the training in the subfolder saved_models of the folder outputs.

**The DistilBERT model**

The DistilBERT folder contains the train_distilbert.py that is used to retrieve the pre-trained version of the model and train it on our new data. There is then the test_distilbert.ipynb file where we test the transformer model on the hold-out sample and the Medium dataset for a true out-of-sample evaluation.

**Other files**

You will find in Desc_stats.ipynb the descriptive statistics and charts that we show in the final report.


