@echo off
echo ============================================================
echo  MACSE634 - Credit Default XAI Project
echo  One-Click Setup and Run
echo ============================================================
echo.

echo [1/3] Installing required Python packages...
pip install lightgbm xgboost shap lime imbalanced-learn scipy scikit-learn pandas numpy matplotlib seaborn openpyxl
echo.

echo [2/3] Packages installed successfully!
echo.

echo [3/3] Running the project...
echo       (This will take 3-8 minutes)
echo       All 20 figures will be saved to: output_figures\
echo.
python credit_default_run.py

echo.
echo ============================================================
echo  Done! Open the output_figures\ folder to see all figures.
echo  Open results_summary.txt for the full output log.
echo ============================================================
pause
