# Programming Problems Classification

A classification system built by a PyTorch model (neural network) used as a basis for recommendation in the context of students taking introductory programming courses, most of the time making use of online judges (like [Codeforces](http://codeforces.com/) or [LeetCode Online Judge](http://leetcode.com/)).

## Description

In this project, it was implemented a classification model to mesure the difficulty (one possible set of class) of programming problems, facilitating recommendation generelly useful for students of introductory programming courses using Online Judge Systems. Comparing with human classification, automatic classification can provide fewer resources (conjunto de especialists or time of classification) and less subjectivity. It was used a fine-tuning model with BERT.

## Demonstration

To see a demo of the model with your own data, acess the [Colab](inferencia.ipynb), donwloading the weigths [here](https://drive.google.com/drive/folders/1OIBKc-g8RIjwpQFieasvxnSfstUvvMT_?usp=drive_link).

## Training

To retrain the model, generating your own weights, follow this steps:

1. Download the repository
2. Acess the directory `workspace` (with the code)
3. Execute the command `python3 train.py`
