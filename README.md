# [Chubb PS2 - Retainly] - Local Setup Guide

## Overview

This document provides step-by-step instructions to set up and run this project on your local machine for development and testing purposes.

---

## Prerequisites

Before you begin, ensure you have the following software installed on your system.

* **Python**: Version 3.13 or higher  
    ```bash
    python --version 
    # Or on some systems:
    python3 --version
    ```
* **Git**  
    ```bash
    git --version
    ```
* **pip**: Python's package installer

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd <PROJECT_FOLDER_NAME>
```

### 2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate 
#### On Windows use `venv\Scripts\activate`

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables 

**GEMINI_API_KEY**="YOUR_GEMINI_API_KEY"
**MEDIASTACK_API_KEY**="YOUR_MEDIASTACK_API_KEY"

### 5. Run the Application
```bash
python app.py
```

## Video Link:
https://drive.google.com/file/d/1EyoNU0Z9q8Gltf6HUcFt5sdvV8bmM2TI/view?usp=sharing

## Dataset Link:
https://drive.google.com/drive/folders/1tuefd734bJxHMcDXZneeJClwciLiFh9f?usp=sharing

### Note:
The dataset should be present in dataset/archive/...csv in root directory of project.
