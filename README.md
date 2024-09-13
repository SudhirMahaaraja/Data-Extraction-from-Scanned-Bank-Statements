# Data-Extraction-from-Scanned-Bank-Statements

## Project Overview
This project focuses on extracting and processing financial transaction data from scanned bank statements in both image and PDF formats. It utilizes a combination of Optical Character Recognition (OCR), Natural Language Processing (NLP), and heuristic-based methods to generate structured data such as dates, transaction details, debit/credit amounts, balances, and transaction modes.

### Key Components:
* Image Preprocessing: Enhances image quality by reducing noise, correcting skew, and normalizing images.
* OCR: Converts text from images and PDFs into machine-readable text using EasyOCR and PyTesseract.
* NLP: Extracts important named entities such as customer names using Spacy's French language model.
* Transaction Parsing: Identifies and extracts transaction details (date, amount, mode of transaction) through regex patterns and context-based analysis.
* Data Cleaning: Handles missing values, removes unnecessary information, and normalizes the output.
* Balance Calculation: Computes running balances based on debit and credit information.
* File Output: Generates output files in both CSV and JSON formats with well-structured transaction data.

### Features
* Image Processing: Prepares scanned images by reducing noise and correcting for skew before text extraction.
* Table Parsing: Extracts tabular data from PDFs using Camelot.
* Named Entity Recognition (NER): Uses Spacy to detect and extract customer names.
* Transaction Mode Detection: Classifies transactions like NEFT, UPI, ATM, etc., using regular expressions.
* Data Extraction: Extracts structured data from unstructured text, including dates, transaction descriptions, amounts, and balances.
* Data Cleaning & Preprocessing: Cleans the extracted data, fills missing values, and removes redundant entries.
* CSV/JSON Output: Saves the cleaned and structured data in CSV and JSON formats for further use.

#### Requirements
To run this project, you'll need the following dependencies installed in your environment:

#### Python Libraries
You can install the required Python packages using the following command:
pip install opencv-python pytesseract pandas camelot-py spacy easyocr deskew scipy
