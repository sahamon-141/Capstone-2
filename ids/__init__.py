"""
IoT Sentinel: a three-module intrusion detection and response pipeline on CICIoT2023.

    Module 1  (ids.module1_detection)   Is this traffic window anomalous?  (IAT + fuzzy logic)
    Module 2  (ids.module2_classification) Which attack is it?              (Random Forest + INT8 edge model)
    Module 3  (ids.module3_mitigation)  What do we do about it?            (playbook + verification)

ids.pipeline ties the three together; app.py is the Streamlit frontend.
"""
