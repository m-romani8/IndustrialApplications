# ClimaSense: Intelligent In-Cabin Climate Control
<div style="text-align:center">
<img src="./Images/ClimaSense logo.png">
</div>
**ClimaSense** is an innovative project aimed at developing a fully automated in-car air conditioning system. The core idea is to create a smart climate control that requires minimal to no driver intervention beyond an initial activation, ensuring optimal comfort and a safer driving experience.

## About The Project

Modern vehicles offer sophisticated climate control, but often still require manual adjustments to maintain a comfortable cabin environment. ClimaSense aims to change this by:

*   **Learning and Adapting:** Intelligently understanding environmental conditions and driver/passenger preferences.
*   **Full Automation:** Managing temperature, fan speed, air distribution, and recirculation autonomously.
*   **Simplicity:** Requiring only a single button press from the driver to activate the intelligent system.
*   **Enhanced Focus:** Reducing distractions by eliminating the need for fiddling with A/C controls.

## Prototype Feature: LSTM-based Driver Drowsiness Detection

As a key additional feature and part of our current prototyping efforts, we are integrating a driver drowsiness detection system. This system is designed to enhance safety by alerting or taking action if the driver shows signs of fatigue.
<div style="text-align:center">
<img src="./Images/Raspberry logo.png">
</div>

### How it Works:

Our prototype utilizes a **Long Short-Term Memory (LSTM) neural network** to analyze driver alertness.
1.  **Input Data:** The LSTM model processes a sequence of **60 consecutive Eye Aspect Ratio (EAR) values**. EAR is a common metric derived from facial landmarks to quantify eye opening.
2.  **Analysis:** By observing the pattern of EAR values over this window, the model learns to identify patterns indicative of drowsiness (e.g., prolonged eye closures, increased blink duration/frequency).
3.  **Output:** The system classifies the driver's state as either alert or drowsy.
4.  **Integration with ClimaSense:** If drowsiness is detected, the ClimaSense system could, for example, proactively adjust the cabin environment (e.g., by introducing cooler, fresh air or changing airflow patterns) to help increase driver alertness, complementing other potential warning systems.

This feature leverages the power of deep learning to add a crucial layer of safety to the automated comfort provided by ClimaSense.

## Technologies (Prototype Focus)

*   **Python:** For model development and system logic.
*   **PyTorch:** For building and training the LSTM model.
*   **OpenCV:** For potential image processing to extract EAR values (though the current focus is on the LSTM with pre-calculated EAR).
*   **Raspberry Pi:** As a potential embedded platform for running the prototype system.

## Goals

*   To create a seamless and truly "set-and-forget" car climate control system.
*   To enhance driver safety through intelligent features like drowsiness detection.
*   To explore the application of machine learning in improving in-vehicle experience.


---

We believe ClimaSense can significantly improve the driving experience by intelligently managing the cabin environment and proactively contributing to driver safety.