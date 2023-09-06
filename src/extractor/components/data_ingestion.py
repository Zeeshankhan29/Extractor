from src.extractor.entity import DataIngestionConfig
from src.extractor import logging
from pathlib import Path
from paddleocr import PaddleOCR,draw_ocr
import spacy.displacy as displacy
import numpy as np
import pandas as pd
import pdfplumber
import os
import sys
import cv2
import shutil
import PyPDF2
import re
import csv
import openpyxl
import streamlit as st
import spacy



class DataIngestion:
    def __init__(self, DataIngestionConfig =  DataIngestionConfig):
        self.dataingestion = DataIngestionConfig
        self.curr_dir = Path(os.getcwd())
        self.data_dir = self.dataingestion.data_dir
        self.ner_model = spacy.load("en_core_web_sm")
        
        
    def normal_image(im_file_path):
        img = cv2.imread(im_file_path)
        cv2.imshow('image',img)
        cv2.waitKey(0)



    def gray_bw_img(im_file_path):
        img = cv2.imread(im_file_path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray, bw_img = cv2.threshold(gray,140,300,cv2.THRESH_BINARY)
        cv2.imshow('image',bw_img)
        cv2.waitKey(0)
        return bw_img


    def noise_removal_bw(im_file_path):
        import numpy as np
        img = cv2.imread(im_file_path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray, bw_img = cv2.threshold(gray,140,300,cv2.THRESH_BINARY)
        kernel = np.ones((2,2),np.uint8)
        image = cv2.dilate(bw_img,kernel,iterations=1)
        kernel = np.ones((2,2),np.uint8)    
        image = cv2.erode(image,kernel,iterations=1)
        image = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
        image = cv2.medianBlur(image,3)
        cv2.imshow('image',image)
        cv2.waitKey(0)
        
        output_folder = 'No_noise'
        os.makedirs(output_folder, exist_ok=True)
        
        # Extract the original filename without extension
        file_name_without_extension = os.path.splitext(os.path.basename(im_file_path))[0]
        
        output_file_path = os.path.join(output_folder, f'{file_name_without_extension}_no_noise.jpg')
        cv2.imwrite(output_file_path, image)
        
        return image

    def jpg_file1(self, img_path):
        img = cv2.imdecode(np.frombuffer(img_path.read(), np.uint8), 1)
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        result = ocr.ocr(img, cls=True)

        # Extract text from the result
        extracted_text = " ".join([line[1][0] for line in result[0]])
        print(extracted_text)
        doc = self.ner_model(extracted_text)
        for data in doc.ents:
            print(data.text,data.start,data.end,data.label_)

        displacy.serve(doc,style='ent')

        exit(0)
        # Extract item names and prices using spaCy NER
        doc = self.ner_model(extracted_text)
        items_and_prices = []
        
        item_name = ""
        price = ""
        items_and_prices =[]
        for token in doc:
            if token.ent_type_ == "PRODUCT":
                item_name = token.text
            elif token.ent_type_ == "MONEY":
                price = token.text
                if item_name:  # Check if there is a valid item name
                    items_and_prices.append({"item": item_name, "price": price})
                    item_name = ""  
        # Create DataFrame and save files
        file_name = Path(img_path.name)
        file1 = file_name.stem
        df = pd.DataFrame(items_and_prices)
        csv_file_name = Path(os.path.join(self.curr_dir, self.data_dir, f'{file1}.csv'))
        df.to_csv(csv_file_name, index=False)
        excel_file_name = Path(os.path.join(self.curr_dir, self.data_dir, f'{file1}.xlsx'))
        df.to_excel(excel_file_name, index=False)

    def jpg_file(self,img_path):
        img = cv2.imdecode(np.frombuffer(img_path.read(), np.uint8), 1)
        from paddleocr import PaddleOCR,draw_ocr
        ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
        # img_path = 'MicrosoftTeams-image (18).png'
        result = ocr.ocr(img, cls=True)
        file_name = Path(img_path.name)
        file1 = file_name.stem
        result = result[0]
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        print(txts)
        scores = [line[1][1] for line in result]
        file_name = Path(os.path.join(self.curr_dir,self.data_dir,f'{file1}.txt'))
        with open(file_name,'w+') as f:
            for i in txts:
                f.write(i+'\n')

         # Creating a DataFrame from the extracted text
        df = pd.DataFrame({'Text': txts})

        # Saving DataFrame to CSV
        csv_file_name = Path(os.path.join(self.curr_dir, self.data_dir, f'{file1}.csv'))
        df.to_csv(csv_file_name, index=False)

        # Saving DataFrame to Excel
        excel_file_name = Path(os.path.join(self.curr_dir, self.data_dir, f'{file1}.xlsx'))
        df.to_excel(excel_file_name, index=False)


    def pdf_file(self, uploaded_file):
        extracted_text = ""

        with uploaded_file as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                extracted_text += page.extract_text()

        file_name = Path(uploaded_file.name)
        file1 = file_name.stem

    # Rest of your processing code (creating Excel workbook, populating sheet, saving Excel file, etc.)



        data_list=[]
        for i in extracted_text.split('\n'):
            if i.endswith('/-'):
                data_list.append(i) 





        # Create a new Excel workbook
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = file1
        # Write the header row
        sheet["A1"] = "Item Name"
        sheet["B1"] = "Price(s)"

        # Populate the Excel sheet with data
        row_num = 2  # Start from the second row
        for item in data_list:
            # Extract item name and prices using regex
            match = re.match(r'^(.+?)\s+((?:\d+/-\s*)+)', item)
            if match:
                item_name = match.group(1)
                prices = match.group(2)

                sheet.cell(row=row_num, column=1, value=item_name)
                sheet.cell(row=row_num, column=2, value=prices)

                row_num += 1

        # Save the Excel file
        excel_path = Path(os.path.join(self.curr_dir, self.data_dir, f'{file1}.xlsx'))
        workbook.save(excel_path)


        print(f"Data saved to '{excel_path}'")


    def text_extraction(self, img_path):
        _, img_extension = os.path.splitext(img_path.name)

        if img_extension.lower() == '.pdf':
            self.pdf_file(img_path)
        elif img_extension.lower() in ['.jpg', '.jpeg', '.png']:
            self.jpg_file(img_path)
        else:
            print("Unsupported image format")


    def run_streamlit(self):
        st.title("Image and PDF Processing")

        uploaded_file = st.file_uploader("Choose an image or PDF file", type=["jpg", "jpeg", "png", "pdf"])

        if uploaded_file is not None:
            st.write("File uploaded successfully!")

            file_extension = uploaded_file.name.split('.')[-1].lower()
            self.text_extraction(uploaded_file)

            st.write("Download Processed Files:")
            if file_extension == "pdf":
                download_link = self.generate_download_button(uploaded_file, "xlsx")
            elif file_extension in ["jpg", "jpeg", "png"]:
                # download_link_csv = self.generate_download_button(uploaded_file, "csv")
                download_link_xlsx = self.generate_download_button(uploaded_file, "xlsx")
                # download_link_txt = self.generate_download_button(uploaded_file, "txt")
                # st.markdown(download_link_csv, unsafe_allow_html=True)
                st.markdown(download_link_xlsx, unsafe_allow_html=True)
                # st.markdown(download_link_txt, unsafe_allow_html=True)
            else:
                st.write("Unsupported file format for download")
                st.markdown(download_link, unsafe_allow_html=True)


    def generate_download_button(self, uploaded_file, file_extension):
        file_name = uploaded_file.name.split('.')[0]
        processed_file_path = Path(self.curr_dir) / self.data_dir / f"{file_name}.{file_extension}"
        return st.download_button(
            label=f"Download {file_name}.{file_extension}",
            data=processed_file_path.read_bytes(),
            file_name=f"{file_name}.{file_extension}"
        )