# Social Implications of NLP-Driven Healthcare Technologies: Empowering Patients through Medicine

## Project Overview
This project focuses on developing a disease and drug recommendation system leveraging NLP techniques to assist clinicians in providing personalized treatment recommendations. The system aggregates and analyzes comprehensive medical histories of patients to facilitate accurate diagnoses, insurance policies, and treatment strategies.

## Folder Structure
- **Data**: Contains fine-tuning train and test CSV files used for model training and evaluation.
- **ClinicalBERT**: Includes scripts and models related to ClinicalBERT, a bidirectional transformer model pre-trained on clinical notes.
- **Customized-BERT**: Contains the custom pre-trained BERT model fine-tuned for disease and drug prediction tasks.
- **Pre-Trained-BERT**: Includes scripts and configurations for using pre-trained BERT models.
- **Rating Prediction**: Contains models and scripts for predicting drug ratings based on user reviews.
- **Disease-prediction-only**: Includes models and scripts focused solely on predicting diseases from patient data.
- **Drug Prediction only**: Contains models and scripts dedicated to predicting drugs based on disease and symptom information.

## Data Sources
The dataset was compiled from multiple sources, including:
- **Mayo Clinic**: Detailed information on symptoms, causes, risk factors, complications, and prevention strategies for diseases and conditions.
- **FAERS**: FDA Adverse Event Reporting System data integrated from various sources, including SNOMED, ICD10, MedDRA, CDF, and UMLS.
- **SympGAN**: Information about diseases, their associated symptoms, corresponding genes, and disease explanations.
- **SemMedDB**: Data on oral diseases and their corresponding explanations.
- **DrugCentral**: Information about diseases, transporters, enzymes, and their associated genes.
- **UCML data - Drugs.com**: Comprehensive details about drugs structured according to the Unified Medical Language System.
- **DrugLib**: User reviews and ratings for various drugs.

## Methodology
1. **Data Collection and Preprocessing**: Data was collected from various sources and underwent preprocessing to ensure consistency and quality. Medical terms were annotated manually, and brand names were converted to generic names.
2. **Model Architecture and Training**:
   - **Section 1**: Pre-training using uncased BERT and ClinicalBERT models.
   - **Section 2**: Fine-tuning models for multi-tasking (predicting diseases and drug recommendations simultaneously).
   - **Section 3**: Fine-tuning models for single-tasking (predicting diseases and drugs individually).

## Results
The BERT uncased model showed the best performance among the models tested. However, there were instances of mispredictions that need further scrutiny.

