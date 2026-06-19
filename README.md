# Feature Engineering – NBA Player Longevity Prediction

## Dataset Source

The dataset used for this analysis is `nba-players.csv`, which contains various performance statistics for NBA players. It played (gp), minutes played (min), points (pts), field goals made (fgm), rebounds (reb), assists (ast), steals (stl), blocks (blk), and more.

## Prediction Goal

The primary goal of this project is to predict `target_5yrs`, which indicates whether a player lasted 5 years in the league (1 for yes, 0 for no). This is a binary classification problem, where we aim to build a machine learning model that can accurately forecast a player's longevity based on their initial performance statistics.

## Feature Engineering Workflow

Our feature engineering workflow involved several crucial steps to prepare the raw data for machine learning:

1.  **Loading the Dataset and Defining Target**: The `nba-players.csv` file was loaded into a pandas DataFrame. The `target_5yrs` column was explicitly defined as the dependent variable for prediction.

2.  **Dropping Non-Predictive Columns**: Columns such as `'Unnamed: 0'` (an arbitrary index), `'name'` (player names), and the `target_5yrs` itself were removed from the feature set. These columns either provide no predictive power or could lead to data leakage, compromising the model's integrity.

3.  **Correlation Analysis and Redundancy Reduction**: A correlation matrix was computed to identify highly correlated features (absolute correlation > 0.9). To reduce multicollinearity and simplify the model, one feature from each highly correlated pair was dropped. For example, `min` was removed as `pts` and `fgm` were strongly correlated and considered more direct performance indicators. Similarly, `fga`, `3pa`, `fta`, and `dreb` were removed due to high correlation with `fgm`, `3p_made`, `ftm`, and `reb` respectively.

4.  **Engineering New Composite Features**: To enhance the predictive power of our dataset, two new composite features were created:
    *   **Points Per Minute (PPM)**: Calculated as `pts / min`. This metric normalizes scoring by playing time, providing a rate-based measure of a player's scoring efficiency.
    *   **Efficiency Rating (EFF)**: A simplified Player Efficiency Rating was computed using the formula `(PTS + REB + AST + STL + BLK - ((FGA - FGM) + (FTA - FTM) + TOV)) / GP`. This metric aims to capture a player's overall on-court contribution, balancing positive and negative actions normalized by games played. Appropriate handling for division by zero was included for both PPM and EFF.

5.  **Handling Null Values**: A check for null values in the performance columns was performed. In this particular dataset, no missing values were found. If any were present, a mean imputation strategy would have been applied to numerical columns to ensure the dataset's completeness for machine learning readiness.

This structured approach to feature engineering aims to create a robust and meaningful set of features that can be effectively used to train a machine learning model for predicting NBA player longevity.
