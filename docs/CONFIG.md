# Configuration Guide

This document provides an in-depth explanation of the configuration options, data properties, and training parameters used by the system. The system supports data collection from keyboard, mouse, and gamepad devices and uses an LSTM autoencoder model for tasks such as anomaly detection. Note that **model hyperparameters are tuned automatically in the background** using the Optuna library, allowing the system to optimize parameters like layer count, neuron count, learning rate, and training epochs without manually setting them.

---

## 1. Overview

The system has three primary modes of operation:
- **Data Collection (programMode = 0):** Captures input data from specified devices at defined intervals.
- **Model Training (programMode = 1):** Processes collected data, updates existing models and trains new models with automatic hyperparameter tuning.
- **Live Analysis (programMode = 2):** Uses the trained model to analyze incoming data in real-time and compute anomaly scores.

The configuration parameters are stored in a file called `config.ini`, which is divided into two sections: **General** and **Model**.

---

## 2. Configuration File: `config.ini`

### General Section

These parameters control the high-level operation and data collection aspects:

- **programMode**  
  *Type:* Integer  
  *Description:*  
  - `0` — Data Collection  
  - `1` — Model Training  
  - `2` — Live Analysis  
  *Usage:* Determines which mode the program will run in.

- **pollInterval**  
  *Type:* Integer (milliseconds)  
  *Description:*  
  Defines the time interval between successive polls of the input devices during data collection.  
  *Impact:*  
  - A shorter poll interval results in higher temporal resolution and more data points.
  - A longer poll interval reduces data size and processing load but might miss short-lived events.
  - As you change pollInterval, consider changing windowSize as the scope of sequences is relative to this setting.

- **captureKeyboard**  
  *Type:* Integer (0 or 1)  
  *Description:*  
  Toggle to enable (`1`) or disable (`0`) keyboard data capture.

- **captureMouse**  
  *Type:* Integer (0 or 1)  
  *Description:*  
  Toggle to enable (`1`) or disable (`0`) mouse data capture.

- **captureGamepad**  
  *Type:* Integer (0 or 1)  
  *Description:*  
  Toggle to enable (`1`) or disable (`0`) gamepad data capture.

- **killKey**  
  *Type:* String  
  *Description:*  
  The key that, when pressed, will terminate the program. **Important:** This key must not be included in any device’s whitelist.

### Model Section

These parameters guide the training and feature selection process:

- **windowSize**  
  *Type:* Integer  
  *Description:*  
  The number of consecutive data points (time steps) used to form each training sequence.  
  *Impact:*  
  - **Larger windowSize:** Captures more context and longer-term dependencies, but requires more data and computational power.
  - **Smaller windowSize:** Faster training with less context; may not capture longer patterns adequately.

- **tuningCycles**  
  *Type:* Integer  
  *Description:*  
  The number of hyperparameter tuning cycles performed during model training.  
  *Impact:*  
  More tuning cycles allow the background process (using Optuna) to explore a larger hyperparameter space for an optimal model configuration.  
  **Note:** The hyperparameter tuning process automatically optimizes parameters such as:
  - Number of LSTM layers (`layerCount`)
  - Neuron count per layer (`neuronCount`)
  - Learning rate (`learningRate`)
  - Number of training epochs (`trainingEpochs`)

- **keyboardWhitelist**  
  *Type:* Comma-separated String  
  *Description:*  
  A list of keyboard keys (features) to be used for training. If left empty, all available keys are used by default.  
  **Caution:** Do not include the `killKey` here.

- **mouseWhitelist**  
  *Type:* Comma-separated String  
  *Description:*  
  Specifies which mouse features to capture (e.g., clicks, movement attributes). If left empty, all default mouse features are used.

- **gamepadWhitelist**  
  *Type:* Comma-separated String  
  *Description:*  
  Specifies which gamepad features to capture. If left empty, all default gamepad features are used.

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
     When multiple devices are being analyzed, compare their anomaly score patterns. A spike in one device might be normal for that particular sensor, while a simultaneous increase across devices might indicate a broader issue.
  
5. **Setting Thresholds:**  
   - **Defining Alerts:**  
     Based on historical data and domain-specific requirements, you can define a threshold for what constitutes an anomaly. When the anomaly score exceeds this threshold, it may trigger alerts or further investigation.
  
**REMEMBER TO INSERT EXAMPLE GRAPHS HERE!!!**  
*The graphic should highlight examples of normal behavior (low, stable scores) versus anomalous behavior (sharp spikes or gradually increasing trends) and should include annotations to guide the interpretation of key features in the graph.*

---

## 5. Best Practices for Intelligent Training

- **Data Quality:**  
  Verify that your data is clean and representative. Remove anomalies or outliers where possible to improve training quality.

- **Iterative Refinement:**  
  Start with moderate settings for `windowSize` and `pollInterval`, and adjust based on early training results and model performance.

- **Balanced Feature Selection:**  
  Experiment with different whitelists to determine which features contribute most effectively to the model’s performance.

- **Resource Consideration:**  
  Keep in mind the trade-off between model accuracy and computational cost. More tuning cycles and larger window sizes can improve performance but at a higher resource cost.

- **Continuous Monitoring:**
  Use the anomaly graphs (in live analysis mode) to continually assess and refine the model configuration.

---

## 6. Conclusion

This documentation aims to provide you with a thorough understanding of the configuration settings and the rationale behind them. By carefully considering the properties of your training data and the impact of each parameter, you can intelligently fine-tune the system to meet your specific needs. Remember, with automatic hyperparameter tuning in the background, your focus can primarily be on optimizing data quality and feature selection for the best overall performance.

For further details, consult the inline comments within the source code and the [Optuna documentation](https://optuna.org/) for more on hyperparameter optimization.

---