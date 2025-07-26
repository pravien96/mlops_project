# MLOps Housing Price Predictor

A complete MLOps pipeline for predicting California housing prices using machine learning best practices.

## 🏗️ Project Overview

This project implements a production-ready machine learning pipeline that:
- Processes California housing data and trains predictive models
- Tracks experiments using MLflow
- Serves predictions via a FastAPI REST API
- Uses Docker for containerization
- Implements CI/CD with GitHub Actions
- Includes comprehensive logging and monitoring

## 🛠️ Technology Stack

- **Python 3.9** - Core programming language
- **FastAPI** - REST API framework
- **scikit-learn** - Machine learning models
- **MLflow** - Experiment tracking and model registry
- **Docker** - Containerization
- **GitHub Actions** - CI/CD pipeline
- **SQLite** - Logging database
- **Prometheus** - Metrics collection (optional)

## 📁 Project Structure

```
mlops-housing-project/
├── .github/workflows/           # CI/CD pipeline
│   └── mlops-pipeline.yml
├── src/                        # Source code
│   ├── data/                   # Data processing
│   │   └── data_processing.py
│   ├── models/                 # Model training
│   │   └── train.py
│   ├── api/                    # FastAPI application
│   │   └── main.py
│   └── middleware/             # Request logging
│       └── prediction_logging.py
├── data/                       # Data storage
│   ├── raw/                    # Raw datasets
│   └── processed/              # Processed datasets
├── models/                     # Trained models
├── logs/                       # Application logs
├── tests/                      # Unit tests
├── docker/                     # Docker configuration
│   └── Dockerfile
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker Desktop
- Git

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd mlops-housing-project
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv mlops-env
   source mlops-env/bin/activate  # On Windows: mlops-env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Process data and train models**
   ```bash
   python src/data/data_processing.py
   python src/models/train.py
   ```

5. **Start the API server**
   ```bash
   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Access the application**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health
   - MLflow UI: `mlflow ui --host 127.0.0.1 --port 5000`

## 🐳 Docker Deployment

### Build and Run Container

```bash
# Build Docker image
docker build -f docker/Dockerfile -t housing-predictor .

# Run container
docker run -p 8000:8000 housing-predictor

# Run with volume mounts (for development)
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  housing-predictor
```

## 📊 API Endpoints

### Core Endpoints

- `GET /` - Welcome message and API info
- `GET /health` - Health check endpoint
- `POST /predict` - Make housing price predictions
- `GET /prediction-history` - View recent predictions
- `GET /metrics` - Prometheus metrics (if enabled)

### Example Prediction Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "MedInc": 8.0,
       "HouseAge": 10.0,
       "AveRooms": 6.0,
       "AveBedrms": 1.2,
       "Population": 3000.0,
       "AveOccup": 3.0,
       "Latitude": 34.0,
       "Longitude": -118.0
     }'
```

### Example Response

```json
{
  "predicted_price": 4.18,
  "price_in_hundreds_of_thousands": "$418.0k",
  "features_used": {
    "MedInc": 8.0,
    "HouseAge": 10.0,
    "AveRooms": 6.0,
    "AveBedrms": 1.2,
    "Population": 3000.0,
    "AveOccup": 3.0,
    "Latitude": 34.0,
    "Longitude": -118.0
  }
}
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/ --cov-report=html
```

## 📈 Experiment Tracking with MLflow

The project uses MLflow for experiment tracking and model management:

1. **View experiments**: `mlflow ui`
2. **Track metrics**: R², MSE, MAE for different models
3. **Model registry**: Best models are automatically registered
4. **Artifact storage**: Models and preprocessors are versioned

### MLflow Tracking Features

- **Experiment comparison**: Compare RandomForest vs Linear Regression
- **Parameter logging**: All hyperparameters are tracked
- **Metric visualization**: Performance metrics over time
- **Model versioning**: Automatic model registration for production

## 📝 Logging and Monitoring

### Application Logging

- **Prediction logs**: All requests and responses logged to SQLite
- **Error tracking**: Comprehensive error logging with stack traces
- **Performance metrics**: Response times and throughput monitoring

### Log Files

- `logs/predictions.db` - SQLite database with prediction history
- Application logs are written to stdout (captured by Docker)

### Monitoring Endpoints

- `/health` - Service health status
- `/prediction-history` - Recent prediction analytics
- `/metrics` - Prometheus-compatible metrics (optional)

## 🔄 CI/CD Pipeline

The GitHub Actions pipeline includes:

### Pipeline Stages

1. **Code Quality**
   - Linting with flake8
   - Code formatting with black
   - Type checking

2. **Testing**
   - Unit tests with pytest
   - Coverage reporting
   - Integration tests

3. **Model Training**
   - Data processing validation
   - Model training pipeline
   - Performance validation

4. **Docker Build**
   - Multi-stage Docker build
   - Security scanning
   - Image optimization

5. **Deployment**
   - Automated deployment to staging
   - Production deployment on manual approval

### Environment Variables

Set these secrets in your GitHub repository:

- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password

## 🏗️ Architecture Overview

### Data Flow

1. **Data Ingestion**: California housing dataset loaded and validated
2. **Preprocessing**: Feature scaling, train/test splitting
3. **Model Training**: Multiple algorithms compared via MLflow
4. **Model Selection**: Best performing model registered
5. **API Serving**: FastAPI serves predictions from registered model
6. **Logging**: All requests/responses logged for monitoring

### Component Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│   Data Layer    │───▶│ Model Layer  │───▶│ API Layer   │
│                 │    │              │    │             │
│ • Raw Data      │    │ • Training   │    │ • FastAPI   │
│ • Preprocessing │    │ • MLflow     │    │ • Validation│
│ • Feature Eng   │    │ • Model Reg  │    │ • Logging   │
└─────────────────┘    └──────────────┘    └─────────────┘
         │                       │                  │
         ▼                       ▼                  ▼
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│ Storage Layer   │    │ Tracking     │    │ Deployment  │
│                 │    │              │    │             │
│ • SQLite Logs   │    │ • MLflow UI  │    │ • Docker    │
│ • Model Files   │    │ • Experiments│    │ • CI/CD     │
│ • Artifacts     │    │ • Metrics    │    │ • Monitoring│
└─────────────────┘    └──────────────┘    └─────────────┘
```

## 🛠️ Development Guidelines

### Code Quality

- **PEP 8**: Follow Python style guidelines
- **Type hints**: Use type annotations where possible
- **Documentation**: Comprehensive docstrings
- **Testing**: Maintain >80% test coverage

### Git Workflow

1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes and test locally
3. Commit with descriptive messages
4. Push and create pull request
5. CI/CD pipeline runs automatically
6. Merge after review and tests pass

## 🐛 Troubleshooting

### Common Issues

**Model Loading Error**
```
Error loading model: [Errno 2] No such file or directory: 'models/best_model.pkl'
```
*Solution*: Run `python src/models/train.py` to train models first.

**Docker Build Fails**
```
Cannot connect to the Docker daemon
```
*Solution*: Ensure Docker Desktop is running and you have permissions.

**Import Errors**
```
ModuleNotFoundError: No module named 'src'
```
*Solution*: Run from project root directory and ensure `__init__.py` files exist.

### Getting Help

1. Check the logs: `docker logs <container-name>`
2. Verify environment: `python --version`, `pip list`
3. Test endpoints: Use `/health` to verify service status
4. Review MLflow: Check experiment tracking for model issues

## 📋 Assignment Completion

This project satisfies all MLOps assignment requirements:

- ✅ **Repository Setup**: Clean directory structure with version control
- ✅ **Data Processing**: Automated data preprocessing pipeline
- ✅ **Model Development**: Multiple models with MLflow tracking
- ✅ **API Packaging**: FastAPI with Docker containerization
- ✅ **CI/CD Pipeline**: GitHub Actions with automated testing
- ✅ **Logging & Monitoring**: Request logging and health monitoring

## 📚 References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Docker Documentation](https://docs.docker.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

## 📄 License

This project is created for educational purposes as part of an MLOps assignment.

---

**Author**: Pravien Madhavan  
**Date**: July 2025  
**Course**: MLOps Project Assignment
