## Issue

1. Over-separating daily ticks in months makes the size of parquet file too small (30 rows, 5 columns). This cause significant requesting costs from S3.
2. Using symbol (e.g. AAPL) as the tag contains the risk of missing same symbol securities. 

## Possible solutions

1. Combine all historical daily ticks data of a single stock into one parquet file
2. Use my own designed security ID as the identifier instead of using symbols

## Your task

- Evaluate the possible solutions. Make a plan to improve and implement it. 
- Implement considering backward compatible, including storage/ and update/ sections
- Write tests to verify your implementation