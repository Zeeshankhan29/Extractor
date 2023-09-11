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
import fitz
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
        self.file_dir = self.dataingestion.file_dir
        
        
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


    def jpg_file(self,img_path):
        img = cv2.imdecode(np.frombuffer(img_path.read(), np.uint8), 1)
        from paddleocr import PaddleOCR,draw_ocr
        ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
        # img_path = 'MicrosoftTeams-image (18).png'
        result = ocr.ocr(img, cls=True)
        file_name = Path(img_path.name)
        file1 = file_name.stem
        print(file1)
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

        df1 = pd.read_csv(csv_file_name)
        values=[]
        for data in df1.values:
            if data[0].isdigit()==False:
                data[0]= data[0].replace('/-', '').replace('Rs', '').replace('/', '')
                values.append(data[0])
            else:
                values.append(data[0])

        first_valid=[]
        for item in range(0,(len(values)-1)):
        #     print((values[item],values[item+1]))
            first_valid.append((values[item],values[item+1]))
        
        final_valid = []
        for index_value in range(len(first_valid) - 1):
            first_item = first_valid[index_value][0]
            second_item = first_valid[index_value][1]
            if (isinstance(first_item, str) and not first_item.isdigit()) and (isinstance(second_item, str) and all(char.isdigit() or char.isspace() for char in second_item)):
                final_valid.append((first_item, second_item))
        df2 = pd.DataFrame(final_valid, columns=['Item Name', 'Price'])   
        valid_path = Path(os.path.join(self.curr_dir,self.file_dir,f'{file1}.csv')) 
        df2.to_csv(valid_path,index=False)  
        # Saving DataFrame to Excel
        excel_file_name = Path(os.path.join(self.curr_dir, self.file_dir, f'{file1}.xlsx'))
        df2.to_excel(excel_file_name, index=False)


    def pdf_file(self, uploaded_file):
        try:
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
                    i=i.replace('/-', '').replace('Rs', '').replace('/', '').replace('.','')
                    data_list.append(i) 

            # Apply the function to the 'Text' column and create new columns
            if data_list == []:
                st.write('Pdf is not editable please try with jpg file format instead')
                exit(0)
            else:       
                df = pd.DataFrame({'Text':data_list})
                # Saving DataFrame to CSV
                csv_file_name = Path(os.path.join(self.curr_dir, self.data_dir, f'{file1}.csv'))
                df.to_csv(csv_file_name, index=False)
                
                print(data_list)
                menu_items = []

                for item in data_list:
                    match = re.search(r'^(.*?)\s+(\d+)$', item)
                    if match:
                        menu_name = match.group(1).strip()
                        price = match.group(2).strip()
                        menu_items.append({'menu_name': menu_name, 'price': price})

                # Now, menu_items is a list of dictionaries, where each dictionary contains 'menu_name' and 'price' keys.
                # You can access the data like this:

                df1= pd.DataFrame(menu_items)
                print(df1)
                # Saving DataFrame to CSV
                csv_file_name = Path(os.path.join(self.curr_dir, self.file_dir, f'{file1}.xlsx'))
                df1.to_excel(csv_file_name, index=False)
        
        except Exception as e:
            print(e)




    def text_extraction(self, img_path):
        _, img_extension = os.path.splitext(img_path.name)

        if img_extension.lower() == '.pdf':
            self.pdf_file(img_path)
        if img_extension.lower() in ['.jpg', '.jpeg', '.png']:
            self.jpg_file(img_path)
        else:
            print("Unsupported image format")



    def run_streamlit(self):
        st.title("Image and PDF Processing")

        uploaded_file = st.file_uploader("Choose an image or PDF file", type=["jpg", "jpeg", "png", "pdf"])

        if uploaded_file is not None:
            try:
                st.write("File uploaded successfully!")
                st.write("Wait to Download the excel file!")
                file_extension = uploaded_file.name.split('.')[-1].lower()
                self.text_extraction(uploaded_file)
                st.write("Click below to download the excel file")
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
            except:
                st.write('Something went wrong with file download try again with different file formats ')


    def generate_download_button(self, uploaded_file, file_extension):
        file_name = uploaded_file.name.split('.')[0]
        processed_file_path = Path(self.curr_dir) / self.file_dir / f"{file_name}.{file_extension}"
        return st.download_button(
            label=f"Download {file_name}.{file_extension}",
            data=processed_file_path.read_bytes(),
            file_name=f"{file_name}.{file_extension}"
        )