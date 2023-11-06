# Remaining-Useful-Life Prediction on turbofan engine with Explainable AI
RUL prediction is performed on the NASA turbofan dataset: https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
The framework used in this project, LSTM-TCN-xAI, is unexplored as of now for this application. The proposed ensemble model's results are compared with standalone LSTM and TCN.
## Results
The evaluation metrics used in this paper are as follows:
RMSE= $`sqrt{Σ (Yi - Ŷi)² / n}`$
R^2= 1- RSS/ TSS
where RSS is the sum of squares of residuals and TSS is the total sum of squares.
MAE: Mean absolute error
MAE=  $`Σ |Yi - Ŷi| / n`$
# Prediction Results
|Method          |RMSE	|MAE   |R-squared |
|----------------|------|------|----------|
|CNN-LSTM        |17.471|	12.45|	0.821   |
TCN	             |14.309|10.505|  0.8805  |
Proposed TCN-LSTM|14.291|10.55 |	0.8807  |

Results of the proposed method using only last test window (out of 5 windows) for the FD003 NASA C-MAPSS dataset:
 ![Actual vs Predicted values for RUL](https://github.com/jo355/dash-rul-prediction/blob/main/Actual%20vs%20Prediction-last%20sample%20for%20each%20engine.png)




