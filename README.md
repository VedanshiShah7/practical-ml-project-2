
# AI Agents for Sepsis EHR Analysis

This project explores the integration of AI agents into Electronic Health Record (EHR) analysis for early prediction and explanation of Sepsis mortality risk. It combines deep learning techniques with a LangChain-based conversational agent to assist healthcare providers in data-driven decision-making.

## 🚀 Project Overview

Sepsis is a life-threatening condition that requires early detection for effective treatment. Our project proposes an AI-powered conversational system that:
- Loads and preprocesses patient EHR data.
- Predicts Sepsis mortality using both traditional ML and deep learning models.
- Provides interpretable predictions using SHAP-based explanations.
- Is deployed as an interactive web application with natural language interaction.

## 🧠 Key Components

- **Data Loader Tool**: Allows CSV upload and basic summary statistics.
- **Preprocessing Tool**: Handles imputation and filtering of missing values.
- **Prediction Tool**: Uses a Decision Tree and a Deep Neural Network (DNN) to predict mortality.
- **Explanation Tool**: Applies SHAP to explain predictions.
- **LangChain Agent**: Interfaces with all tools via natural language prompts.
- **Frontend (React App)**: Enables interactive user input and visualization.

## 📁 Directory Structure

```
project-root/
│
├── frontend/                # React-based web interface
├── main.py                 # LangChain agent and tool integration
├── model.py                # DNN model architecture
├── predict.py              # Prediction tool using ML/DNN
├── preprocessing.py        # Imputation and filtering logic
├── explain.py              # SHAP-based explanation module
├── sepsis.py               # Shared helper functions
└── README.md               # You're here!
```

## 🛠️ Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/VedanshiShah7/practical-ml-project-2.git
   cd sepsis-ehr-agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Start the streamlit application
    ```bash
    streamlit run app.py

## 👩‍⚕️ Usage

Interact with the AI agent using natural language questions such as:
- Run imputation model for patient with id 200003 from the file "/Users/vedanshi/Documents/GitHub/practical-ml-project-2/main/data/test.csv" using the preloaded imputation model from path "/Users/vedanshi/Documents/GitHub/practical-ml-project-2/main/OpenManus-main/app/tool/tPatchGNN/tPatchGNN/experiment_48851.ckpt"
- Give me the mortality prediction for patient with id 200003 in the file "/Users/vedanshi/Documents/GitHub/practical-ml-project-2/main/data/test.csv"

## 📚 References

1. Johnson, A. E. W., Pollard, T. J., Shen, L., et al. (2016).  
   MIMIC-III, a freely accessible critical care database.  
   *Scientific Data, 3*, 160035. [https://doi.org/10.1038/sdata.2016.35](https://doi.org/10.1038/sdata.2016.35)

2. Singer, M., Deutschman, C. S., Seymour, C. W., et al. (2016).  
   The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).  
   *JAMA, 315*(8), 801–810. [https://doi.org/10.1001/jama.2016.0287](https://doi.org/10.1001/jama.2016.0287)

3. Desautels, T., Calvert, J., Hoffman, J., et al. (2016).  
   Prediction of sepsis in the ICU using machine learning and physiological data.  
   *BioMed Research International*, 2016. [https://doi.org/10.1155/2016/9308692](https://doi.org/10.1155/2016/9308692)

4. Shickel, B., Tighe, P. J., Bihorac, A., & Rashidi, P. (2017).  
   Deep EHR: A survey of recent advances in deep learning techniques for electronic health record (EHR) analysis.  
   *IEEE Journal of Biomedical and Health Informatics, 22*(5), 1589–1604. [https://doi.org/10.1109/JBHI.2017.2767063](https://doi.org/10.1109/JBHI.2017.2767063)

5. Johnson, A. E. W., Ghassemi, M. M., Nemati, S., et al. (2017).  
   Machine learning and decision support in critical care.  
   *Proceedings of the IEEE, 104*(2), 444–466. [https://doi.org/10.1109/JPROC.2015.2501978](https://doi.org/10.1109/JPROC.2015.2501978)

6. Futoma, J., Morris, J., & Lucas, J. (2015).  
   A comparison of models for predicting early hospital readmissions.  
   *Journal of Biomedical Informatics, 56*, 229–238. [https://doi.org/10.1016/j.jbi.2015.05.016](https://doi.org/10.1016/j.jbi.2015.05.016)

7. Goldstein, B. A., Navar, A. M., Pencina, M. J., & Ioannidis, J. P. A. (2017).  
   Opportunities and challenges in developing risk prediction models with electronic health records data: A systematic review.  
   *Journal of the American Medical Informatics Association, 24*(1), 198–208. [https://doi.org/10.1093/jamia/ocw042](https://doi.org/10.1093/jamia/ocw042)

8. Beam, A. L., & Kohane, I. S. (2018).  
   Big data and machine learning in health care.  
   *JAMA, 319*(13), 1317–1318. [https://doi.org/10.1001/jama.2017.18391](https://doi.org/10.1001/jama.2017.18391)

9. Rajkomar, A., Dean, J., & Kohane, I. (2019).  
   Machine learning in medicine.  
   *New England Journal of Medicine, 380*, 1347–1358. [https://doi.org/10.1056/NEJMra1814259](https://doi.org/10.1056/NEJMra1814259)

10. Nemati, S., Holder, A., Razmi, F., et al. (2018).  
    An Interpretable Machine Learning Model for Accurate Prediction of Sepsis in the ICU.  
    *Critical Care Medicine, 46*(4), 547–553. [https://doi.org/10.1097/CCM.0000000000002936](https://doi.org/10.1097/CCM.0000000000002936)

11. Zhao, Y., Liu, C., Zhang, H., et al. (2024).  
    Irregular Multivariate Time Series Forecasting: A Transformable Patching Graph Neural Networks Approach.  
    *Proceedings of the 41st International Conference on Machine Learning (ICML 2024).*  
    GitHub: [https://github.com/usail-hkust/t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN)

12. Manna and Poem Lab. (2024).  
    OpenManus: Modular AI Agent Framework for Clinical and Document Analysis.  
    GitHub Repository: [https://github.com/mannaandpoem/OpenManus](https://github.com/mannaandpoem/OpenManus)


## View full report [here](https://docs.google.com/document/d/1RtC93yPIw8yH2gbDPiGU0s51s2EClknmpqLDXZLwH9s/edit?usp=sharing)


## 👨‍💻 Authors
Vedanshi Shah

Brandeis University - COSI 149B (Spring 2025)
