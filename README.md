# Automated Proctoring for Online Exams
# ESE440/441 Senior-Design by Aaron Lin and Hang Chen
---

## Description
This program was created for a year long project with the potential to ease the process professors' take when grading online exams. With the pandemic, everyone had to resort to taking exams online, and with the realization of its convenience, it may continue to be a popular way of taking exams in the future. Upon running the program, the GUI will be displayed in the browser, where inputs, such as video recordings, can be uploaded for the program to process. On completion of processing, the results from the program will be shown in the form of graphs and tables. The users can make use of the generated data with their own judgement.

## Note: Google Cloud Platform is required to run this program. The following APIs are utilized in the program, and will need to be enabled :
- Google Cloud Speech API
- Google Vision API

---
## To Run This Program:
1. Create a virtual environment
2. Clone this repository
3. Run **pip install -r requirements.txt** to install all dependencies
4. Create two folders with names **annotated_json** and **image_samples**
5. Download the required files for **models**, **resources**, and **yolov3-config** folders
6. Replace the **.json** file in the source code with your Google Cloud Credential json file. This json file needs to be placed in **resources** folder
7. Replace all Google Cloud related urls with desired urls

---
## Folder Explanations
- **annotated_json** is used to store the json files that contain the results from Google Vision API
- **image_samples** is used to store image frames from the video
- **models** is used to store the yolov3 models and fastai model(no longer necessary in current implementation)
- **resources** is used to store Google Cloud Credential, student id photo and recorded video file
- **yolov3-config** is used to store yolov3 configuration files
