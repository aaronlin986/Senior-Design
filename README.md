# ESE440/441 Senior-Design by Aaron Lin and Hang Chen
---

## Note: A Google Cloud Platform account is required to run this program. Google Cloud Storage and following APIs need to be enabled in Google Cloud Platform as well
- Google Cloud Speech API
- Google Vision API

---
## To run this program:
1. Create a virtual environment
2. Clone this repository
3. Run **pip install -r requirements.txt** to install all dependencies
4. Create two folder with name **annotated_json** and **image_samples**
5. Download the required files for **models**, **resources** and **yolov3-config** folder
6. Replace the **.json** file in the source code with your Google Cloud Credential json file. This json file need to be placed in **resources** folder
7. Replace all Google Cloud related urls with desired urls

---
## Folder explanations
- **annotated_json** is used to store the json files that contain the results from Google Vision API
- **image_samples** is used to store image frames from the video
- **models** is used to store the yolov3 models and fastai model(no longer necessary in current implementation)
- **resources** is used to store Google Cloud Credential, student id photo and recorded video file
- **yolov3-config** is used to store yolov3 configuration files
---

### If you have any questions, please reach me via my email in my GitHub profile page.
