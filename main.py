import cv2
import pytesseract
import pandas as pd
import camelot
import json
import re
import os
import numpy as np
import spacy
import easyocr
from deskew import determine_skew
from scipy.ndimage import rotate

# Load French NLP model for Named Entity Recognition (NER)
nlp = spacy.load("fr_core_news_md")

# Function to preprocess images: noise reduction, skew correction, normalization
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction using Gaussian blur
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    # Skew correction
    skew_angle = determine_skew(denoised)
    if abs(skew_angle) > 0.5:  # Skew correction only if necessary
        denoised = rotate(denoised, skew_angle, reshape=False)

    # Normalization
    norm_img = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    return norm_img

# Function to extract text from images using EasyOCR (Deep Learning-based OCR)
def extract_text_from_image(image_path):
    reader = easyocr.Reader(['fr'], gpu=True)  # French language support
    img = preprocess_image(image_path)

    if img is None:
        return ""

    result = reader.readtext(img, detail=0, paragraph=True)
    return ' '.join(result)

# Function to extract tables from PDFs using Camelot
def extract_tables_from_pdf(pdf_path):
    tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')  # Changed to 'stream' to handle irregular tables
    extracted_data = []
    for table in tables:
        df = table.df
        extracted_data.append(df)
    return extracted_data

# Helper function to clean and normalize text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove excessive spaces
    text = text.replace('\n', ' ')
    text = text.strip()
    return text

# Function to extract mode of transaction using extended patterns and contextual analysis
def extract_mode_of_transaction(text):
    patterns = [
        r'\b(transfer|virement|cash|cheque|check|card|payment|online|NEFT|IMPS|UPI|RTGS|POS|ATM|debit|credit)\b',
        r'\b(deposit|withdrawal|auto-debit|auto-credit|loan payment|refund)\b'
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group().capitalize()

    mode_map = {
        "ATM": "ATM",
        "POS": "POS",
        "NEFT": "NEFT",
        "IMPS": "IMPS",
        "UPI": "UPI",
        "RTGS": "RTGS",
        "Cheque": "Cheque",
        "Check": "Cheque",
        "Online": "Online"
    }

    for key in mode_map:
        if key in text:
            return mode_map[key]

    return "Unknown"

# Function to extract customer name using NLP (Named Entity Recognition)
def extract_customer_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "Unknown"

# Function to parse balance and balance amount
def calculate_balance(debits, credits):
    balance = []
    current_balance = 0

    for debit, credit in zip(debits, credits):
        if debit:
            current_balance -= float(debit.replace(',', '').replace(' ', ''))
        elif credit:
            current_balance += float(credit.replace(',', '').replace(' ', ''))
        balance.append(current_balance)

    return balance

# Function to parse text using regex and heuristics
def parse_text_with_heuristics(text):
    date_pattern = r'\b(\d{2}[./-]\d{2}[./-]\d{4})\b'
    amount_pattern = r'\b(\d{1,3}(?:[\s.,]\d{3})*(?:[\s.,]\d{2})?)\b'
    bank_name_pattern = r'(Banque|Crédit\s*|Nom\s*de\s*la\s*Banque)\s*[:\s]*([\w\s]+)'

    dates = re.findall(date_pattern, text)
    amounts = re.findall(amount_pattern, text)

    doc = nlp(text)
    labels = []
    bank_names = []
    mode_of_transaction = extract_mode_of_transaction(text)
    customer_name = extract_customer_name(text)

    for ent in doc.ents:
        if ent.label_ == "ORG":
            bank_names.append(ent.text)
        else:
            labels.append(ent.text)

    bank_name = bank_names[0] if bank_names else "Banque Inconnue"

    debits = []
    credits = []
    for amount in amounts:
        if "Débit" in text:
            debits.append(amount)
        else:
            credits.append(amount)

    max_len = max(len(dates), len(labels), len(debits), len(credits))
    dates.extend([''] * (max_len - len(dates)))
    labels.extend([''] * (max_len - len(labels)))
    debits.extend([''] * (max_len - len(debits)))
    credits.extend([''] * (max_len - len(credits)))

    balance_list = calculate_balance(debits, credits)

    parsed_data = []
    for i in range(max_len):
        row = {
            'date': dates[i].strip() if i < len(dates) else '',
            'Debit': debits[i].strip() if i < len(debits) else '',
            'credit': credits[i].strip() if i < len(credits) else '',
            'amount': debits[i] if i < len(debits) else (credits[i] if i < len(credits) else ''),
            #'balance': balance_list[i] if i < len(balance_list) else '',
            'mode of transaction': mode_of_transaction,
            #'customer name': customer_name,
            'bank name': bank_name.strip(),
            'transaction details': labels[i].strip() if i < len(labels) else '',
            'balance amount': balance_list[i] if i < len(balance_list) else '',
        }
        if any(row.values()):
            parsed_data.append(row)

    return parsed_data

# Function to save parsed data to CSV and JSON with specific column names
def save_data_to_files(parsed_data, output_csv, output_json):
    df = pd.DataFrame(parsed_data,
                      columns=['date', 'Debit', 'credit', 'amount', 'balance', 'mode of transaction', 'customer name',
                               'bank name', 'transaction details', 'balance amount'])
    df.to_csv(output_csv, index=False, encoding='utf-8')
    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(parsed_data, json_file, indent=4, ensure_ascii=False)

# Function to process images and PDFs in a folder
def process_folder(folder_path):
    parsed_data = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.lower().endswith((".jpg", ".png")):
            text = extract_text_from_image(file_path)
            clean_text_content = clean_text(text)
            parsed_data.extend(parse_text_with_heuristics(clean_text_content))
        elif file_name.lower().endswith(".pdf"):
            tables = extract_tables_from_pdf(file_path)
            for table in tables:
                text = ' '.join(table.values.flatten())
                clean_text_content = clean_text(text)
                parsed_data.extend(parse_text_with_heuristics(clean_text_content))
    return parsed_data

if __name__ == "__main__":
    folder_path = "data"  # Folder containing the image and PDF files
    output_csv = "output.csv"
    output_json = "output.json"

    extracted_data = process_folder(folder_path)

    # Save extracted and parsed data
    save_data_to_files(extracted_data, output_csv, output_json)

    print(f"Data saved to {output_csv} and {output_json}.")

# Function to process CSV for handling missing values and cleaning
def process_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Display the first few rows of the dataframe for initial inspection
    print("Initial Data:")
    print(df.head())

    # Handling missing values
    df.fillna('Unknown', inplace=True)  # Replace NaN with 'Unknown' for all columns

    # Removing rows with irrelevant values (if necessary)
    df = df[~(
        (df['date'] == 'Unknown') &
        (df['Debit'] == 'Unknown') &
        (df['credit'] == 'Unknown') &
        (df['amount'] == 'Unknown') &
        (df['balance'] == 'Unknown') &
        (df['mode of transaction'] == 'Unknown') &
        (df['customer name'] == 'Unknown') &
        (df['bank name'] == 'Unknown') &
        (df['transaction details'] == 'Unknown')
    )]

    # Additional cleaning: Remove rows with only empty or 'Unknown' values in certain key columns
    df = df[df[['date', 'amount', 'balance']].notna().any(axis=1)]

    # Saving cleaned data to a new CSV file
    df.to_csv(output_csv, index=False, encoding='utf-8')

    print(f"Cleaned data saved to {output_csv}.")

if __name__ == "__main__":
    input_csv = "output.csv"
    cleaned_csv = "cleaned_output.csv"

    process_csv(input_csv, cleaned_csv)
    print(f"CSV file processed and cleaned.")
