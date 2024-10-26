
# Breast Cancer Classification Using Machine Learning

This project focuses on classifying breast cancer using machine learning techniques. Follow the instructions below to set up and run the application.

![Breast Cancer Classification](https://github.com/Addisu-Amare/fse4ai_team4_project/blob/main/Assets/app.jpg)
![Docker Run](https://github.com/Addisu-Amare/fse4ai_team4_project/blob/main/Assets/docker_run.jpg)

## Prerequisites

Make sure you have the following installed:
- **Git**: 
- **Docker**:
- **make**

## Setup Instructions

1. **Create a Directory:**
   ```bash
   mkdir breast_cancer_classification
   ```

2. **Change Directory:**
   ```bash
   cd breast_cancer_classification
   ```

3. **Clone the Repository:**
   ```bash
   git clone https://github.com/Addisu-Amare/fse4ai_team4_project
   ```

4. **Navigate to the Project Directory:**
   ```bash
   cd fse4ai_team4_project
   ```

5. **Build the Docker Image:**
   ```bash
   make build
   ```

6. **Run the Application:**
   ```bash
   make run
   ```

## Project Structure

```
breast_cancer_classifier/
├── Dockerfile                     # Dockerfile for building the image
├── Makefile                       # Makefile for managing builds and runs
├── requirements.txt               # Python dependencies
├── app.py                         # Flask application
├── preprocessing.py               # Data preprocessing script
├── training.py                    # Model training script
├── postprocessing.py              # Model evaluation script
├── breast_cancer_detector.pickle   # Trained model file
├── templates/                     # Directory for HTML templates
│   └── index.html                 # Main HTML template for the app
└── static/                        # Directory for static files (CSS, images)
    ├── css/                       # CSS files
    │   └── style.css              # Stylesheet for the app
    └── img/                       # Image files
        ├── skoltech.jpg              # Background image
```

## Usage
1. Open your web browser and navigate to `http://localhost:5000`.
2. Fill in the input fields with the required features related to breast cancer.
3. Click on "Click Here to Predict" to get the prediction.
4. The result will be displayed below the form.

## Makefile Targets

- `make build`: Builds the Docker image.
- `make run`: Runs the Flask application in a Docker container.
- `make preprocess`: Runs only the preprocessing stage.
- `make train`: Runs only the training stage.
- `make postprocess`: Runs only the postprocessing stage.
- `make clean`: Cleans up any stopped containers.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project uses data from [CANCER.NET](https://www.cancer.net).
- Special thanks to all contributors and resources that helped in developing this application.
```

### Explanation of Sections

1. **Project Title**: A brief title describing what the project is about.
2. **Prerequisites**: Lists software requirements needed to run the application.
3. **Setup Instructions**: Step-by-step guide on how to set up and run the project locally using Docker.
4. **Project Structure**: A tree view of your project's directory structure, which helps users understand how files are organized.
5. **Usage**: Instructions on how to use the web application once it's running.
6. **Makefile Targets**: Information on available commands in your Makefile for building and running specific stages of your application.
7. **License**: Information about licensing (you can adjust this based on your actual license).
8. **Acknowledgments**: Credits or references to external resources or contributors.

### Conclusion
This `README.md` provides a comprehensive overview of Breast Cancer Classification App .If you have any more questions or need additional features implemented, feel free to ask!
Citations:
[1] https://www.youtube.com/watch?v=WGNI-k20GNo
[2] https://philpep.org/blog/a-makefile-for-your-dockerfiles/
[3] https://docs.docker.com/build/building/multi-stage/
[4] https://stackoverflow.com/questions/51253987/building-a-multi-stage-dockerfile-with-target-flag-builds-all-stages-instead-o
[5] https://docs.docker.com/build/concepts/context/
[6] https://www.kaggle.com/code/akshat0007/breast-cancer-classification-using-knn-and-svm
