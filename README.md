# 🏭 Predictive Maintenance AI 🧪 🚀

**Predictive Maintenance using XGBoost, tracked with MLflow, versioned with DVC, and deployed on AWS Fargate.**

![App Screenshot](assets/app_screenshot.png)

## 🏗️ Architecture

```mermaid
graph TD
    subgraph DVC Pipeline
        A[Ingestion Script] -->|src.data.ingestion| B(data/raw/predictive_maintenance.csv)
        B -->|src.data.preprocessing| C[Preprocessing Script]
        C --> D(data/processed/train.csv)
        C --> E(data/processed/test.csv)
        D -->|src.pipelines.training| F[Training Pipeline]
        F --> G(models/xgboost_model.pkl)
        F --> H(models/feature_engineer.pkl)
        E -->|src.pipelines.evaluation| I[Evaluation Pipeline]
        G --> I
        I --> J[Metrics & Confusion Matrix]
    end

    subgraph MLflow
        F -.->|Log Params/Metrics| K{MLflow Tracking Server}
        I -.->|Log Metrics| K
    end

    subgraph Deployment
        G -->|Build| L[Docker Container]
        H -->|Build| L
        L -->|Push| M[AWS ECR]
        M -->|Deploy| N[AWS ECS Fargate]
        N -.->|Serve| Q[FastAPI Backend]
    end

    subgraph User Interface
        Q --> O[Gradio App]
        O --> P((End User))
    end

    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style F fill:#f9f,stroke:#333
    style I fill:#f9f,stroke:#333
    style K fill:#bbf,stroke:#333
    style N fill:#bfb,stroke:#333
    style Q fill:#bfb,stroke:#333
```

## 🛠️ Tech Stack

-   **Experiment Tracking**: ![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)
-   **Data Versioning**: ![DVC](https://img.shields.io/badge/DVC-945DD6?style=flat&logo=dvc&logoColor=white) + **AWS S3**
-   **Modeling**: ![XGBoost](https://img.shields.io/badge/XGBoost-EB9924?style=flat&logo=xgboost&logoColor=white) ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
-   **API**: ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
-   **Containerization**: ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
-   **Cloud Ops**: ![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat&logo=amazon-aws&logoColor=white) (ECR Registry, ECS Fargate Compute)
-   **App**: ![Gradio](https://img.shields.io/badge/Gradio-FD6F00?style=flat&logo=gradio&logoColor=white)

## 📂 Project Structure

```text
predictive-maintenance/
├── .dvc/                  # DVC configuration
├── .dvcignore
├── .github/
│   └── workflows/
│       └── deploy.yml     # CI/CD Pipeline
├── app/
│   └── gradio_app.py      # Frontend Application
├── api/                   # FastAPI Backend
├── assets/
│   └── app_screenshot.png
├── data/                  # Data directory (tracked by DVC)
├── docker/
│   ├── Dockerfile         # Container definition
│   └── task-definition.json
├── mlruns/                # MLflow tracking data
├── models/                # Saved models (tracked by DVC/MLflow)
├── notebooks/             # Jupyter notebooks for EDA and experiments
├── scripts/               # Utility scripts
├── src/                   # Source code
│   ├── data/
│   ├── features/
│   └── pipelines/
├── tests/                 # Unit and integration tests
├── dvc.yaml               # DVC pipeline definition
├── dvc.lock               # DVC lock file
└── requirements.txt       # Python dependencies
```

## ⚡ Installation & Usage

### 1. Setup Environment
```bash
# Clone the repo
git clone https://github.com/aymanz12/predictive-maintenance.git
cd predictive-maintenance

# Install dependencies
pip install -r requirements.txt
```

### 2. Fetch Data & Models
Pull the latest versioned data and model artifacts from AWS S3 using DVC.
```bash
dvc pull
```

### 3. Track Experiments
Launch the MLflow UI to view training runs and metrics.
```bash
mlflow ui
# Access at http://localhost:5000
```

### 4. Run the Application
Start the Gradio dashboard locally.
```bash
python app/gradio_app.py
# Access at http://localhost:7860
```

## ☁️ AWS Deployment Guide

This project is configured for continuous deployment. However, you can manually trigger the workflow steps:

1.  **Build Docker Image**
    ```bash
    docker build -t predictive-maintenance -f docker/Dockerfile .
    ```

2.  **Push to AWS ECR**
    ```bash
    # Login to ECR
    aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<region>.amazonaws.com
    
    # Tag and Push
    docker tag predictive-maintenance:latest <aws_account_id>.dkr.ecr.<region>.amazonaws.com/predictive-maintenance:latest
    docker push <aws_account_id>.dkr.ecr.<region>.amazonaws.com/predictive-maintenance:latest
    ```

3.  **Update ECS Service**
    ```bash
    aws ecs update-service --cluster <cluster_name> --service <service_name> --force-new-deployment
    ```
