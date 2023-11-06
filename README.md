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
### Prediction Results
|Method          |RMSE	|MAE   |R-squared |
|----------------|------|------|----------|
|CNN-LSTM        |17.471|	12.45|	0.821   |
TCN	             |14.309|10.505|  0.8805  |
Proposed TCN-LSTM|14.291|10.55 |	0.8807  |

Results of the proposed method using only last test window (out of 5 windows) for the FD003 NASA C-MAPSS dataset:
 ![Actual vs Predicted values for RUL](https://github.com/jo355/dash-rul-prediction/blob/main/Actual%20vs%20Prediction-last%20sample%20for%20each%20engine.png)

### xAI SHAP Interpretations
For gaining insight into the proposed model’s predictions, the SHAP explainer was trained on 100 samples of the train dataset and computed SHAP values for the first 30 samples of test data (i.e., sample 1’s engine unit number is 1, sample 2’s is 2 and so on). Generally, the colors of the values represent the contributions of features. Red value means that the feature increases the model’s RUL value while the blue denotes that the feature decreases the model’s RUL value.

Figure 1 depicts the decision plot for the last window in test engine number 29 in FD004. 
•	The decision plot shows the contribution of sensors for a prediction in decreasing order of importance. For engine 30, T50, total temperature at LPT outlet, followed by phi (ratio of fuel flow to Ps30), P30(total pressure at HPC outlet) and Ps30(static pressure at HPC outlet) are the sensors with the largest contribution. 
•	The plot is centred on the base value which is the average of the model over the test dataset. It lies on the x-axis, here, the base RUL value of the TCN-LSTM model’s predictions for the first 30 engines in the test dataset is 90.5 cycles approx. 
•	The predicted RUL value is indicated at the top where the line strikes at i.e., 92.3 cycles approximately and this determines the color of the line, i.e., red line here since the predicted value is higher than the actual model value.
•	The shapely values of each sensor from the bottom to the top are added cumulatively to the base value of the model to produce the final output value. We can observe here, that sensor W32 , LPT coolant bleed, is the only sensor that slightly reduces( subtracts) the base RUL value while the others only add on , thereby leading to the increase in the output RUL.
![Decision Plot of the Proposed TCN-LSTM on the last window of data (window number =30) of the test engine unit no. = 29 with actual RUL value= 89 cycles](https://github.com/jo355/dash-rul-prediction/blob/main/engine_29_last_window.png)

Figure 2 shows a force plot representing the same sample as Figure 1. 
•	The force plot clearly depicts the range and direction of the impact of contribution (positively or negatively). For example, sensor T50, decreases the value of the output RUL from the base value the most while phi increases the base value the most. 
•	Another advantage is showing the precise predicted value which is 92.34 cycles.
•	A disadvantage of the force plot is that all features are not clearly listed. For instance, like figure 1, figure 2 also shows that P50 followed by phi are the highest contributors. However, it fails to show the lowest contributors (sensor W31 and sensor T30), stopping the plot at sensor Nf, thereby omitting the three least contributing features.
![Sensor explanations provided by SHAP force plot for the last window of data ( window =30) of the test engine unit no. 30 with the actual RUL value at 89 cycles and predicted value is 92.25 cycles](https://github.com/jo355/dash-rul-prediction/blob/main/single%20sample%20rul-engine%2029%20last%20window.png)

Figure 3 shows the decision plot of the overall predictions for engine 30, including all 30 test data windows that were generated during pre-processing phase as window length. The blue lines depict the windows of data that outputted a lower RUL than the base RUL( 90.5 cycles), with the lowest predicted RUL being 85 cycles approx. and the largest predicted RUL value as 96 cycles.![Decision plot of the proposed method on the test engine unit no. = 30 for data in all windows (30 in total)](https://github.com/jo355/dash-rul-prediction/blob/main/force_plot_engine_30.jpg)

Figure 4 depicts a bar plot depicting the importance of features w.r.t predictions of engine unit no 29.
![Feature importance bar plot for window=30 in test engine unit no. 29](https://github.com/jo355/dash-rul-prediction/blob/main/feature_importance_engine_30.png)





