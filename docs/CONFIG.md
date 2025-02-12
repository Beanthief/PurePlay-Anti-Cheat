# Configuration Guide

This document provides an in-depth explanation of the configuration options, data properties, and training parameters used by the system. The system supports data collection from keyboard, mouse, and gamepad devices and uses an LSTM autoencoder model for tasks such as anomaly detection. Note that **model hyperparameters are tuned automatically in the background** using the Optuna library, allowing the system to optimize parameters like layer count, neuron count, and learning rate without manually setting them.

---

## 1. Overview

The system has three primary modes of operation:
- **Data Collection:** Captures input data from specified devices at a defined rate.
- **Model Training:** Processes collected data, updates existing models and trains new models with automatic hyperparameter tuning.
- **Live Analysis:** Uses the trained model to analyze incoming data in real-time and compute anomaly scores.

The configuration parameters are stored in a file called `config.ini`, which is divided into five sections: **General**, **Keyboard**, **Mouse**, **Gamepad** and **Model**.

---

## 2. Configuration File: `config.ini`

### General Section

These parameters control the high-level operation of the program.

- **programMode**  
  *Type:* Integer  
  *Description:*  
  - `0` — Data Collection  
  - `1` — Model Training  
  - `2` — Live Analysis  

- **recordBind**  
  *Type:* String  
  *Description:* This bind will toggle data collection. Can be any feature of any device.  
  **Important:** This bind must not be included in any whitelists and cannot be the killKey.

- **killKey**  
  *Type:* String  
  *Description:* The bind that, when pressed, will terminate the program. Must be a keyboard key.  
  **Important:** This bind must not be included in any whitelists and cannot be the killKey.

### Device Sections

These parameters control device and data characteristics.

- **capture(Device)**  
  *Type:* Integer (0 or 1)  
  *Description:* Toggle to enable (`1`) or disable (`0`) data capture on that device.  

- **(device)Whitelist**  
  *Type:* Comma-separated String  
  (OVERWRITTEN BY LOADED MODEL)  
  *Description:*  
  A list of input features to be used for training. If left empty, all available features for that device are used. This setting is ignored in mode 0. Do not include the `killKey` in the keyboardWhitelist.  
  
  **Possible Values:**
  Keyboard:  
  `a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, +, -, *, /, ., ,, <, >, ?, !, @, #, $, %, ^, &, *, (, ), _, =, {, }, [, ], |, \\, :, ;, , , ~, enter, esc, backspace, tab, space, caps lock, num lock, scroll lock, home, end, page up, page down, insert, delete, left, right, up, down, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, print screen, pause, break, windows, menu, right alt, ctrl, left shift, right shift, left windows, left alt, right windows, alt gr, windows, alt, shift, right ctrl, left ctrl`  

  Mouse:  
  Note: angle and magnitude are relative to the previous poll  
  `left, right, middle, x1, x2, angle, magnitude`  

  Gamepad:  
  `DPAD_UP, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT, START, BACK, LEFT_THUMB, RIGHT_THUMB, LEFT_SHOULDER, RIGHT_SHOULDER, A, B, X, Y, LT, RT, LX, LY, RX, RY`

- **pollingRate**  
  *Type:* Integer (hz)  
  (OVERWRITTEN BY LOADED MODEL)  
  *Description:* Defines the rate of device polling. It is not recommended to go over 125hz.  
  *Impact:*  
  - A higher polling rate results in higher temporal resolution and more data points.
  - A lower polling rate reduces data size and processing load but might miss short-lived events.
  - As you change polling rate, consider changing windowSize as the scope of sequences is relative to this setting.

  - **windowSize**  
  *Type:* Integer (>= 5)  
  (OVERWRITTEN BY LOADED MODEL)  
  *Description:* The number of time steps in an input sequence.  
  *Impact:*  
  - **Larger windowSize:** Captures more context and longer-term dependencies, but requires more data and computational power.
  - **Smaller windowSize:** Faster training with less context; may not capture longer patterns adequately.

### Training Section

These parameters guide the training and feature selection process.  
It is recommended that you first attempt to train with the default parameters.

  - **trialEpochs**  
  *Type:* Integer  
  *Description:* The number of epochs per trial.
  *Impact:*  
  - **More epochs:** Increases tuning time, provides slightly more accurate tuning
  - **Less epochs:** Reduces tuning time, provides slightly less accurate tuning

- **tuningTrials**  
  *Type:* Integer  
  *Description:* The number of trials through the automatic tuning process.  
  *Impact:*  
  More tuning trials allows the training loop to explore a larger hyperparameter space for higher accuracy.
  **Note:** The hyperparameter tuning process automatically optimizes parameters such as:
  - Number of LSTM layers (`layerCount`)
  - Neuron count per layer (`neuronCount`)
  - Learning rate (`learningRate`)

  - **finalEpochs**  
  *Type:* Integer  
  *Description:* The number of epochs used to train tuned model.
  *Impact:*  
  - **More epochs:** Increases accuracy and training time, risks overfitting
  - **Less epochs:** Reduces accuracy and training time, risks underfitting

---

## 3. Understanding Feature Selection

### What is Feature Selection?
Feature selection is a critical process in model training. It involves choosing the most relevant input features from your available data (keyboard, mouse, or gamepad) to help the model learn meaningful patterns. Effective feature selection can:
- **Reduce Dimensionality:** By limiting the number of features, the model trains faster and requires fewer computational resources.
- **Minimize Noise:** Selecting only the most pertinent features reduces the risk of the model learning from irrelevant or redundant data.
- **Improve Accuracy:** A carefully chosen set of features helps the model focus on the key signals, leading to better generalization and improved performance on unseen data.

### Best Practices for Feature Selection
- **Leverage Domain Knowledge:**  
  Use insights about your application to select features that are most likely to contribute valuable information. For instance, specific keys or mouse movements might be more indicative of user behavior in your use case. They may also be more prone to cheating inputs such as strafe or anti-recoil macros.
- **Iterative Refinement:**  
  Start with a broad set of features and evaluate the model’s performance. Gradually remove or adjust features based on validation results, focusing on those that have the most impact.
- **Monitor for Overfitting:**  
  Including too many features can lead to overfitting. A lean, well-chosen feature set helps the model generalize better.
- **Reduce Redundancy:**  
  Analyze your features for high correlations. Eliminating redundant or highly correlated features can simplify the training process and improve model stability.

### Impact on Model Accuracy and Performance
- **High-Quality Features:**  
  When the model is trained on carefully selected features, it can more easily capture the essential patterns in the data, leading to improved accuracy and faster convergence.
- **Irrelevant or Excessive Features:**  
  Including features that add little value or introduce excessive noise may hinder the training process, slow down convergence, and negatively affect overall performance.

By thoughtfully curating the set of features used in training, you can achieve a balance between complexity and performance, ensuring that the model is both efficient and accurate.

---

## 4. Interpreting the Anomaly Graph

**Overview:**  
The anomaly graph displays computed anomaly scores over successive time windows. Each line on the graph represents a specific device (keyboard, mouse, or gamepad), with the x-axis indicating the progression of time (or window index) and the y-axis showing the corresponding anomaly score.

**How to Interpret the Graph:**

1. **Normal Behavior:**  
   - **Low and Stable Scores:**  
     Under typical conditions, anomaly scores remain low and relatively stable. This indicates that the current input data aligns well with the patterns learned during training.
  
2. **Anomalous Behavior:**  
   - **Spikes in Anomaly Score:**  
     Sudden peaks or spikes in the graph signal that the system has encountered data that deviates significantly from the norm. These anomalies could indicate:
       - Unexpected user behavior.
       - Unusual input patterns.
       - Potential system malfunctions or external disturbances.
  
3. **Trend Analysis:**  
   - **Gradual Increase:**  
     A slowly rising trend in anomaly scores might suggest a gradual shift in input behavior. While not immediately alarming, this trend should be monitored as it could lead to future anomalies.
   - **Isolated Peaks vs. Consistent Patterns:**  
     Compare isolated spikes to recurring patterns. Isolated anomalies may be transient or due to noise, whereas consistent high anomaly scores across multiple windows could point to systemic issues.

4. **Multi-Device Comparison:**  
   - **Cross-Device Insights:**  
     When multiple devices are being analyzed, compare their anomaly score patterns. A spike in one device might indicate cheating for that particular device, while a simultaneous increase across devices might indicate a broader issue as most cheats operate on one feature or device at a time.
  
5. **Setting Thresholds:**  
   - **Defining Alerts:**  
     Based on historical data and domain-specific requirements, you can define a threshold for what constitutes an anomaly. When the anomaly score exceeds this threshold, it may trigger alerts or further investigation.
  
**REMEMBER TO INSERT EXAMPLE GRAPHS HERE!!!**  
*The graphic should highlight examples of normal behavior (low, stable scores) versus anomalous behavior (sharp spikes or gradually increasing trends) and should include annotations to guide the interpretation of key features in the graph.*

---

## 5. Best Practices for Intelligent Training

- **Data Quality:**  
  Verify that your data is clean and representative. Remove anomalies or outliers where possible to improve training quality. You may want to tie capture states to game states so UI navigation and typing won't interfere.

- **Iterative Refinement:**  
  Start with moderate settings for `windowSize` and `pollingRate`, and adjust based on early training results and model performance.

- **Balanced Feature Selection:**  
  Experiment with different whitelists to determine which features contribute most effectively to the model’s performance.

- **Resource Consideration:**  
  Keep in mind the trade-off between model accuracy and computational cost. More tuning trials and larger window sizes can improve performance but at a higher resource cost.

- **Continuous Monitoring:**
  Use the anomaly graphs (in live analysis mode) to continually assess and refine the model configuration.

---

## 6. Conclusion

This documentation aims to provide you with a thorough understanding of the configuration settings and the rationale behind them. By carefully considering the properties of your training data and the impact of each parameter, you can intelligently fine-tune the system to meet your specific needs. Remember, with automatic hyperparameter tuning in the background, your focus can primarily be on optimizing data quality and feature selection for the best overall performance.

For further details, consult the inline comments within the source code and the [Optuna documentation](https://optuna.readthedocs.io/en/stable/index.html) for more on hyperparameter optimization.

---