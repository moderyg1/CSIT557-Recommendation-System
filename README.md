# CSIT 557 Recommendation System

## Members: Garrett Modery, Emilio Herrera

## Dataset: CiaoDVD
*CiaoDVD is a dataset crawled from the entire category of DVDs from the dvd.ciao.co.uk website in December, 2013*

Rows: 72665

Columns: 6 (userId, movieId, movie-categoryId, reviewId, movieRating, reviewDate)

## Dependencies
Dependencies can be installed with pip install -r requirements.txt

## Running the Code
Each notebook in the ./notebooks folder can be run for its designated purpose.

eda.ipynb - Preliminary exploratory data analysis to visualize the data

ItemBasedCF.ipynb - Item Based Collaborative Filtering which outputs the RMSE and MAE errors for cosine and pearson similarities

UserBasedCF.ipynb - User Based Collaborative Filtering which outputs the RMSE and MAE errors for cosine and pearson similarities

SVD.ipynb - Singular Value Decomposition which outputs the RMSE and MAE errors

comparisons.ipynb - Charts and tables comparing each method

## Results Summary

|   Method   |   MAE    |   RMSE  |   Model       |
 -----------  ---------- ---------  --------------
|   Cosine   |   0.819  |   1.083 |   User Based  |
|   Pearson  |   0.843  |   1.103 |   User Based  |
|   Cosine   |   0.829  |   1.102 |   Item Based  |
|   Pearson  |   0.847  |   1.117 |   Item Based  |
|   SVD      |   0.740  |   0.958 |   SVD         |