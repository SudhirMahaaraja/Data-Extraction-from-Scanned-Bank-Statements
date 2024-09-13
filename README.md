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
```pip install opencv-python pytesseract pandas camelot-py spacy easyocr deskew scipy```

#### Language Models
To process French texts, the Spacy language model is required. Install it with:
```python -m spacy download fr_core_news_md```

#### System Dependencies
Tesseract OCR: Required for text extraction from images.
```https://github.com/tesseract-ocr/tesseract```

Ghostscript: Needed for Camelot to process PDFs.
```https://www.ghostscript.com/releases/index.html```

## How It Works
1. Preprocessing Images
preprocess_image(image_path): Converts input images to grayscale, reduces noise, corrects skew, and normalizes the image. This prepares the image for more accurate OCR extraction.
2. OCR on Images
extract_text_from_image(image_path): Uses EasyOCR (with French language support) to read text from preprocessed images. The output is a string that can be further processed.
3. Table Extraction from PDFs
extract_tables_from_pdf(pdf_path): Uses Camelot to extract tables from PDF files. The extracted tables are converted into Pandas DataFrames for easy manipulation.
4. Text Parsing & Information Extraction
parse_text_with_heuristics(text): Parses raw text extracted from OCR or tables. It extracts dates, transaction details, debit/credit amounts, and modes of transaction. Uses regular expressions and NLP to identify important entities.
5. Balance Calculation
calculate_balance(debits, credits): Calculates a running balance by processing the list of debit and credit amounts extracted from the text.
6. Saving Data
save_data_to_files(parsed_data, output_csv, output_json): Saves the extracted and processed data to CSV and JSON files.
7. Processing Folders
process_folder(folder_path): Automates the extraction of data from all image and PDF files within a folder. Extracts text from images using EasyOCR, and from PDFs using Camelot.
8. Cleaning CSV Data
process_csv(input_csv, output_csv): Processes the generated CSV file by handling missing values and removing rows with irrelevant or incomplete data.
