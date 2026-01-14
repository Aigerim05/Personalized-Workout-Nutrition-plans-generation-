## Full Project Report

A detailed walkthrough of the recommendation pipeline, data analysis, formulas for feature engineering, clustering and model training results, and further suggested improvements are documented here:
**[Report.pdf](./Report.pdf)**

## Actuality

Many users fail to achieve long-term progress even with access to workout apps and online diet plans. One of the reasons is that current solutions typically rely on **generic** and standard plans, leading to disengagement, poor progress, and high dropout rates. The second reason is that workout and diet plans are not used in **combination** which leads to low effectiveness in achieving the fitness goal.  

## Project Summary

This project responds to this problem by tailoring recommendations to each **user’s goals** and **body profile**, and generating workout and meal plans that align with their **target calorie** loss or gain.
It was developed as the final exam project for the Data Mining course (Fall 2025).

This project:

* learns user groups through **K-Means clustering**,
* predicts exercise and meal suitability using **XGBoost models**,
* applies content-based filtering **(TF-IDF + cosine similarity)** and **MMR reranking**,
* **balances calories**, macronutrients, and training structure,
* and outputs a complete **day-by-day** plan tailored to weight loss, maintenance, or gain.

## Input and Output 

Final workout and nutrition plans can be found here: **[Workout&Nutrition.pdf](./plan.pdf)**

### Input: User-Provided Features

* Age
* Gender
* Height, weight
* Goal: lose / maintain / gain
* Goal duration (days)
* Activity level (Beginner, Intermediate, Advanced)
* Workout frequency availability

### Output: Personalized PDF Plan

The system produces a **day-by-day plan**, including:

* **Daily meals** with calories, macros, portion adjustments
* **Workout sessions** grouped by training-day type (push/pull/legs/core)
* **Target vs actual calorie balance**, summarized over the entire period
* **Estimated weight change**, based on energy intake/expenditure

## Tools, Methods & Concepts

Python (Pandas, scikit-learn, XGBoost), K-Means clustering, Linear Regression baseline, TF-IDF vectorization, cosine similarity, MMR reranking, outlier handling (Winsorization), one-hot encoding, standard scaling, RandomizedSearchCV.

