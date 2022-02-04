# Analyzing Data regarding Stress, Depression and Anxiety with clustering

## About the project
The purpose of this project is to explore and give a brief analysis of a dataset containing answers to a Stress, Depression and Anxiety test. This analysisis also tries to explore the relationship between these mental health issues with the personality traits of an individual and also their provided general profile.

All of the exploration and results are commented in the pdf that comes alongside this project (everything is written in Spanish).

## Tools
Python3 alongside the libraries of pandas, matplotlib, scipy, sklearn, numpy.

## Methods
In order to perform this exploration, these are the following data mining techniques/methods employed: K-means clustering, dendrograms, the elbow method for finding an optimal *k*, frequent patterns, association rules and Normalized Mutual Information between clusterings.

## About the dataset
The dataset used for this project comes from Kaggle, it's in the following link: https://www.kaggle.com/yamqwe/depression-anxiety-stress-scales

As a brief description, these are the types of information that were used in order to make this project: 

* Answers to a questionary whose purpose is to calculate levels of depression, anxiety and stress (DASS)
* Answers to a questionary that helps to determine the personality traits of a person.
* General data about the one who's taking the test (native language, age, gender, education level, etc.).

## Description of files

* **Report.pdf**: the formal analysis of the exploration that was done.
* **Experiments.ipynb**: Jupyter Notebook where all the obtained results from the report can be replicated.
* **DashApp.py**: a Python script that when executed launches a Dash app that renders on the browser. This one contains some interactable graphs for having a general idea of how the dataset works. It also contains a K-means clustering visualization over the first two principal components for some of the results that were commented in the report.
