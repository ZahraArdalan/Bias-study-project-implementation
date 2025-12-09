================================================================================
                          BIAS STUDY IN AI MODELS PROJECT
================================================================================

📚 ABOUT THE RESEARCH ARTICLE
-------------------------------------------------------------------------------
This project implements concepts from the scientific article:

"Where, why, and how is bias learned in medical image analysis models?"
Published in The Lancet EBioMedicine (2024)

Article Authors: Emma A. M. Stanley, Raissa Souza, Matthias Wilms, Nils D. Forkert
University of Calgary, Canada

📌 WHAT THE ARTICLE STUDIES:
-------------------------------------------------------------------------------
The original research investigates how deep learning models used for medical
image analysis (like MRI, CT scans) can learn "shortcuts" from biased data.

Real-world problem: When AI models are trained on medical images, they might
learn to recognize:
- Hospital scanner type instead of actual disease
- Patient demographics instead of pathology
- Imaging artifacts instead of real medical conditions

This leads to "algorithmic bias" - the AI works well for some groups but
fails for others, even when the actual disease is the same.

🎯 WHAT THIS PROJECT DEMONSTRATES
-------------------------------------------------------------------------------
We create a SIMPLIFIED VERSION of the article's experiment:

1. We generate synthetic data (like simplified MRI measurements)
   - Class 0: "Healthy" patients (values around 5)
   - Class 1: "Disease" patients (values around 10)

2. We add artificial "bias" (like different scanner types):
   - 80% of disease samples have the bias
   - 20% of healthy samples have the bias

3. We train two AI models:
   - MODEL A: Can only see the disease feature (no bias access)
   - MODEL B: Can see both disease feature AND bias feature

4. We analyze the results:
   - Compare accuracy of both models
   - Check if Model B uses bias as a "shortcut"
   - Measure how strongly bias influences predictions

🔬 KEY FINDINGS WE EXPECT TO SEE
-------------------------------------------------------------------------------
1. Model B (with bias access) will have HIGHER accuracy
2. Model B learns to use bias as a shortcut
3. The bias gets encoded in the model's weights
4. This demonstrates "shortcut learning" - exactly what the article discusses!

🛠️ HOW TO RUN THIS PROJECT
-------------------------------------------------------------------------------

STEP 1: SAVE THE FILES
-----------------------
Save this code as a Python file named: bias_project.py

STEP 2: INSTALL REQUIRED SOFTWARE
---------------------------------
You need Python installed. If you don't have it:
1. Go to python.org
2. Download Python 3.8 or higher
3. Install it 

STEP 3: INSTALL PYTHON PACKAGES
--------------------------------
Open Command Prompt (Windows) or Terminal (Mac/Linux):

For Windows:
--------------
1. Press Windows Key + R
2. Type "cmd" and press Enter
3. Type these commands one by one:

   pip install numpy
   pip install matplotlib
   pip install scikit-learn

For Mac/Linux:
---------------
1. Open Terminal
2. Type:

   pip3 install numpy matplotlib scikit-learn

STEP 4: RUN THE PROJECT
------------------------
Navigate to where you saved the file:

cd Desktop  (or your folder location)
python bias_project.py

If "python" doesn't work, try "python3":
python3 bias_project.py

📊 WHAT HAPPENS WHEN YOU RUN IT
-------------------------------------------------------------------------------

CONSOLE OUTPUT:
---------------
You'll see step-by-step execution:
1. Data generation statistics
2. Model training results
3. Bias analysis
4. Performance comparisons

GENERATED FILES (in 'results' folder):
---------------------------------------
1. results.png        - Charts and visualizations
2. results.json       - All numerical results
3. analysis_report.txt - Summary of findings

INTERPRETING THE RESULTS:
-------------------------
Look for these key numbers:

1. ACCURACY COMPARISON:
   - Model A (no bias): around 84-86%
   - Model B (with bias): around 86-88%
   - Difference: Positive number means bias helped

2. BIAS IMPORTANCE:
   - If > 30%: Strong bias influence (shortcut learning)
   - If 15-30%: Moderate bias influence
   - If < 15%: Minimal bias influence

3. REAL-WORLD MEANING:
   - This simple experiment shows how real medical AI can become biased
   - Similar to: AI learning scanner type instead of disease




