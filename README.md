# EV vs. Conventional Car Purchase Prediction and User Profiling

This repository contains a collection of Jupyter Notebooks documenting our project on analyzing social media behavior to predict whether users will purchase an electric vehicle (EV) or a conventional fuel vehicle. The project leverages Large Language Models (LLMs) to extract quantitative signals from Reddit posts, build detailed user profiles, visualize these profiles, and apply various machine learning algorithms for binary classification.

## Project Overview

The goal of this project is to analyze user behavior by harvesting, cleaning, and reasoning over social media posts (primarily from automotive subreddits). The insights extracted by an LLM are used to generate quantitative features describing each user's interests, attitudes, and technical discussions regarding EVs. These features are then used to predict a user's fuel type preference (Electric vs. Conventional).

## Repository Structure

- **02user_harvesting.ipynb**  
  Harvests Reddit posts from various automotive subreddits, filters for relevance, and saves aggregated data in `author_posts.csv`.

- **03user_reasoning.ipynb**  
  Uses LLM-based reasoning on the harvested posts to extract key signals about car ownership, specific models, and fuel types.

- **04user_history.ipynb**  
  Cleans and filters the raw review data to retain posts that indicate car ownership and valid fuel type information.  
  **Output:** `filtered_user_reviews.csv`.

- **05user_profiling.ipynb**  
  Constructs detailed user profiles by processing aggregated post headlines (with timestamps). The LLM generates a concise profile summary (~150 words) and computes six quantitative metrics:
  - interest  
  - attitude  
  - technical expertise  
  - adoption readiness  
  - engagement  
  - communication clarity  
  **Output:** `user_profiles_llm.csv`.

- **06radar_chart.ipynb**  
  Visualizes the quantitative user profiles by generating radar (spider) charts for each user. The charts display the six metrics with minimal clutter (only the user ID is displayed).  
  **Output:** Radar chart images (e.g., `radar_chart_profiles_no_labels.png`).

- **07predicting.ipynb**  
  Uses the six quantitative metrics as features for binary classification of fuel type preferences (Electric vs. Conventional). Multiple classifiers (Logistic Regression, Random Forest, SVC, and KNN) are evaluated on a consistent train-test split, and their performance is compared via evaluation metrics and visualized through heatmaps and tables.  
  **Output:** Model evaluation metrics and comparative visualizations.

## Final Dataset Preparation for Classification

After LLM-based profiling, the user profiles are merged with the filtered reviews to include the target variable (`fuel_type`). This final dataset (`final_dataset.csv`) is used to train and evaluate machine learning models for predicting fuel type preference.

## Experimental Results

Our experimental evaluations using standard classifiers (such as Logistic Regression, Random Forest, SVC, and KNN) indicate accuracies in the range of 60–63%, which are only slightly above a naïve baseline. This suggests that while our LLM-based feature extraction provides some signal, further feature engineering or model tuning may be needed to significantly improve performance.
