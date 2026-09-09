# Facial Tension Monitor

### **Installation steps for testing** 

**1. Download all dependencies** 
   - Go to [Deployment for testing](</Deployment for testing>) folder, then download the [prototype_OpenFace_v2.3.zip](</Deployment for testing/prototype_OpenFace_v2.3.zip?raw=true>) file. Extract the **.zip** file into the local directory. 
   - **Install** [**Python 3.11**](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe).  
     During installation, check **"Add python.exe to PATH"**.  
     To verify, open **cmd** and run:
     ```powershell
        python --version
     ``` 
   - **Download** [**OpenFace**](https://github.com/TadasBaltrusaitis/OpenFace/releases/download/OpenFace_2.2.0/OpenFace_2.2.0_win_x64.zip).  
     Extract it. By default, the script expects it at exactly:
     `C:\OpenFace\OpenFace_2.2.0_win_x64\FeatureExtraction.exe`
   - **Install** [**Visual C++ Redistributable**](https://aka.ms/vs/16/release/vc_redist.x64.exe).
   - Download **OpenFace** model files. From the extracted **OpenFace** folder, there's a `download_models.ps1` script, right-click it and choose **"Run with PowerShell"**. If windows blocks it, right-click the folder then choose **Open in Terminal**, then run:
      ```powershell
         Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
         .\download_models.ps1
      ```
**2. Create a virtual environment and install dependencies**
   - Open the extracted `prototype_OpenFace_v2.3` folder, right-click the folder then choose **Open in Terminal.**  
     Set up the virtual environment by running:
     ```powershell
      py -3.11 -m venv .venv
      .venv\Scripts\Activate.ps1
      python -m pip install -r requirements.txt
      ```

**3. Run the script**
   - Open the extracted `prototype_OpenFace_v2.3` folder, right-click the folder then choose **Open in Terminal.**  
     Then to run the script, run this command: 
     ```powershell
      .venv\Scripts\Activate.ps1
      python prototype_OpenFace_v2_3.py
      ```

---
Dataset used for training can be seen here  
https://zenodo.org/record/1188976
