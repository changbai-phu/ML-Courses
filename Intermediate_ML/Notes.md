Course outline
- tackle data types: missing values, categorical variables
- design pipelines to improve the quality of ML code
- advanced techniques (cross-validation) to validate model
- built models (XGBoost) that are widely used 
- avoid leakage

1. Deal with missing values - 3 approaches
- Drop columns with missing values (may lose a lot of useful info)
- Imputation: fill missing values with some other number, e.g., mean value of the column (may be far off from the real value, or missing itself is also meaningful info)
- An extension to imputation: fill missing values with some number and plus adding extra column to indicate which are the imputated entries.
- 2nd and 3rd perform bettern than 1st approach