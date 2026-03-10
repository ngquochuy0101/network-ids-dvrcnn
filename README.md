# Network Intrusion Detection System

> Production-ready intrusion detection system using DV-RCNN (Dual-View Residual CNN) architecture for NSL-KDD dataset analysis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Performance](#performance)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Development](#development)
- [MLOps Practices](#mlops-practices)
- [Deployment](#deployment)
- [Contributing](#contributing)

---

## Overview

This project implements a **Network Intrusion Detection System (IDS)** using deep learning to classify network traffic into 5 categories:
- **Benign**: Normal traffic
- **DoS**: Denial of Service attacks
- **Probe**: Reconnaissance attacks
- **R2L**: Remote-to-Local attacks
- **U2R**: User-to-Root privilege escalation

The system features:
- **Dual-View CNN Architecture**: Processes both sequential features and correlation matrices
- **Real-time Analysis**: Streamlit web interface for interactive analysis
- **Production-Ready**: Comprehensive logging, error handling, and testing
- **Vietnamese UI**: User-friendly interface in Vietnamese with English code documentation

---

## Key Features

### Model Architecture
- **DV-RCNN** with attention-based fusion
- **1.38M parameters** optimized for NSL-KDD dataset
- **Dual-view processing**: 1D CNN for temporal patterns, 2D CNN for feature correlations
- **Residual connections** for improved gradient flow

### Data Processing
- Automatic preprocessing pipeline (StandardScaler + OneHotEncoder)
- Sliding window generation with configurable parameters
- Correlation matrix image creation (11x11)
- Handles categorical features: protocol_type, service, flag

### Web Interface (Streamlit)
- File upload (.txt, .csv formats)
- Real-time data preview and statistics
- Batch inference with progress tracking
- Interactive visualizations (Plotly)
- CSV export functionality

### Metrics & Evaluation
- **Accuracy, Precision, Recall, F1-Score** (weighted & macro)
- Per-class performance metrics
- Confusion matrix visualization
- Ground truth comparison (when available)

---

## Performance

Evaluation on NSL-KDD Test+ dataset (22,544 samples):

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 82.43% |
| **F1-Score (Weighted)** | 83.05% |
| **F1-Score (Macro)** | 68.17% |
| **Inference Speed** | 2,227 samples/sec |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Benign | 88.52% | 85.79% | 87.14% | 9,711 |
| DoS | 82.83% | 91.70% | 87.04% | 7,458 |
| Probe | 79.42% | 72.01% | 75.53% | 2,421 |
| R2L | 69.40% | 64.55% | 66.89% | 2,887 |
| U2R | 13.51% | 59.70% | 22.03% | 67 |

**Note**: U2R class has low precision due to extreme class imbalance (67 samples vs 9,711 Benign samples).

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster inference

### Installation

1. **Clone the repository**
   ```bash
    git clone https://github.com/ngquochuy0101/network-ids-dvrcnn
    cd network-ids-dvrcnn
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   ```

### Running the Application

**Option 1: Using Streamlit directly**
```bash
streamlit run app.py
```

**Option 2: Using batch script (Windows)**
```bash
run_app.bat
```

The application will open in your default browser at `http://localhost:8501`

### Usage Workflow

1. **Upload Data**: Select NSL-KDD test file (.txt or .csv)
2. **Preview**: Review data statistics and sample records
3. **Analyze**: Click "Phân tích" button to run predictions
4. **Review Results**: View metrics, charts, and confusion matrix
5. **Export**: Download predictions as CSV file

---

## Architecture

### System Architecture

```
┌─────────────────┐
│   User Upload   │
│   (CSV/TXT)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│   Data Preprocessing        │
│   ├─ StandardScaler         │
│   ├─ OneHotEncoder          │
│   └─ Column Transformer     │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Window Creation           │
│   ├─ Sliding Window (L=1)   │
│   ├─ 1D View (D × L)         │
│   └─ 2D View (11×11 corr)   │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   DV-RCNN Model             │
│   ├─ CNN1D Branch           │
│   ├─ CNN2D Branch           │
│   ├─ Attention Fusion       │
│   └─ 3-Layer Classifier     │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Post-processing           │
│   ├─ Label Mapping          │
│   ├─ Metrics Calculation    │
│   └─ Visualization          │
└─────────────────────────────┘
```

### Model Architecture Details

**DV-RCNN (Dual-View Residual CNN)**

```python
Input:
  - x_1d: (batch, d_in, window_len)  # 1D sequential view
  - x_2d: (batch, 1, 11, 11)          # 2D correlation matrix

Branch 1: CNN1D
  - Conv1D(d_in → 96, k=3) + BatchNorm + ReLU + MaxPool
  - Conv1D(96 → 192, k=3) + BatchNorm + ReLU
  - Conv1D(192 → 384, k=3) + BatchNorm + ReLU
  - Residual connection: Conv1D(d_in → 384, k=1)
  - Global MaxPool → Dropout(0.3) → FC(384 → 96)

Branch 2: CNN2D
  - Conv2D(1 → 48, k=3) + BatchNorm + ReLU + MaxPool
  - Conv2D(48 → 96, k=3) + BatchNorm + ReLU
  - Conv2D(96 → 192, k=3) + BatchNorm + ReLU
  - Conv2D(192 → 384, k=3) + BatchNorm + ReLU
  - Residual connection: Conv2D(1 → 384, k=1)
  - Global MaxPool → Dropout(0.2) → FC(384 → 96)

Fusion: Attention Module
  - Concat [h1, h2] → FC(192 → 96) → Tanh
  - Attention weights: FC(96 → 2) → Softmax
  - Weighted fusion → FC(192 → 96)

Classifier:
  - FC(96 → 192) + BatchNorm + GELU + Dropout(0.3)
  - FC(192 → 96) + BatchNorm + GELU + Dropout(0.2)
  - FC(96 → 5)

Output: (batch, 5)  # Logits for 5 classes
```

**Total Parameters**: 1,385,383

---

## Project Structure

```
network-ids/
│
├── app.py                          # Main Streamlit application
├── dvrcnn-best.ipynb              # Model training notebook
├── test_mlops_optimization.py     # Comprehensive test suite
│
├── requirements.txt               # Python dependencies
├── runtime.txt                    # Python version (3.9)
├── .gitignore                     # Git ignore rules
├── README.md                      # This file
│
├── model/
│   └── best_model.pt              # Trained model weights (d_in=121)
│
├── plk/
│   └── preprocessor.pkl           # Fitted preprocessing pipeline
│
├── dataset/
│   ├── KDDTrain+.txt             # Training data (for preprocessor fitting)
│   └── KDDTest+.txt              # Test data (for evaluation)
│
└── docs/
    └── OPTIMIZATION_HISTORY.md    # Development history & optimization logs
```

### Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `app.py` | Streamlit web application | 910 |
| `test_mlops_optimization.py` | Test suite with metrics validation | 520 |
| `dvrcnn-best.ipynb` | Training notebook | N/A |
| `model/best_model.pt` | Trained DV-RCNN model | 5.3 MB |
| `plk/preprocessor.pkl` | Fitted StandardScaler + OneHotEncoder | ~1 MB |

---

## Development

### Setup Development Environment

1. **Clone and create virtual environment**
   ```bash
  git clone https://github.com/ngquochuy0101/network-ids-dvrcnn
  cd network-ids-dvrcnn
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov black flake8  # Additional dev tools
   ```

3. **Run tests**
   ```bash
   python test_mlops_optimization.py
   ```

### Testing

The project includes a comprehensive test suite (`test_mlops_optimization.py`) that validates:

- ✅ Model loading and architecture
- ✅ Preprocessor pipeline
- ✅ Data loading and preprocessing
- ✅ Window creation (1D and 2D views)
- ✅ Inference on sample data
- ✅ Full pipeline on test set (22,544 samples)
- ✅ Metrics calculation (accuracy, precision, recall, F1-score)
- ✅ Performance verification (accuracy ≥ 80%)

**Run tests:**
```bash
python test_mlops_optimization.py
```

**Expected output:**
```
✅ Test 1/10 PASSED: Config initialized correctly
✅ Test 2/10 PASSED: Model loaded successfully (d_in=121)
✅ Test 3/10 PASSED: Preprocessor loaded
...
✅ Test 10/10 PASSED: Accuracy verification (82.43% ≥ 80.00%)

==================== TEST SUMMARY ====================
Total Tests: 10 | Passed: 10 | Failed: 0
Overall Accuracy: 82.43%
F1-Score (Weighted): 83.05% | F1-Score (Macro): 68.17%
Inference Speed: 2,227 samples/sec
Status: ALL TESTS PASSED ✅
```

### Code Quality

The project follows these best practices:

- **Type Hints**: 95% coverage on major functions
- **Docstrings**: 100% coverage (Google style)
- **Logging**: Centralized logging with appropriate levels
- **Error Handling**: Comprehensive try-except blocks
- **Configuration**: Dataclass-based config management
- **Code Style**: PEP 8 compliant

---

## MLOps Practices

### Model Versioning

- Model stored with full architecture and weights
- Checkpoint includes: `model_state_dict`, training metadata
- Version tracking via Git tags and model filename

### Logging & Monitoring

```python
# Application logging
logger.info("Model loaded successfully")
logger.error("Error processing file", exc_info=True)

# Metrics tracked:
- Inference time per batch
- Preprocessing time
- Memory usage
- Prediction distribution
```

### Configuration Management

All hyperparameters and paths centralized in `Config` dataclass:

```python
@dataclass
class Config:
    device: torch.device
    model_path: str = "model/best_model.pt"
    n_classes: int = 5
    window_len: int = 1
    image_h: int = 11
    batch_size: int = 128
    # ... more configs
```

### Error Handling

- Graceful fallback when model/preprocessor not found
- User-friendly error messages in UI
- Detailed logging for debugging

### Testing Strategy

- **Unit tests**: Individual components (preprocessing, window creation)
- **Integration tests**: Full pipeline validation
- **Performance tests**: Accuracy threshold validation (≥80%)
- **Regression tests**: Ensure updates don't break functionality

### Continuous Integration (Future)

Planned CI/CD pipeline:
```yaml
# .github/workflows/test.yml
- Automated testing on push
- Code quality checks (flake8, black)
- Performance regression tests
- Docker image building
```

---

## Deployment

### Option 1: Streamlit Cloud (Recommended)

1. Push repository to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Configure:
   - Python version: 3.9 (via `runtime.txt`)
   - Main file: `app.py`
4. Deploy with one click

**Note**: Ensure `model/best_model.pt` is tracked (update `.gitignore`)

### Option 2: Local Deployment

```bash
# Clone and setup
git clone https://github.com/ngquochuy0101/network-ids-dvrcnn
cd network-ids-dvrcnn
pip install -r requirements.txt

# Run application
streamlit run app.py --server.port 8501
```

### Option 3: Docker (Future)

```dockerfile
# Planned Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Option 4: Cloud Platforms

- **Heroku**: Add `Procfile` and `setup.sh`
- **AWS EC2**: Deploy with nginx + supervisor
- **Azure Web Apps**: Use Python runtime
- **Google Cloud Run**: Containerized deployment

---

## Dataset

### NSL-KDD Dataset

This project uses the **NSL-KDD** dataset, an improved version of KDD Cup 1999:

- **Training Set**: 125,973 records
- **Test Set**: 22,544 records  
- **Features**: 41 (after dropping `num_outbound_cmds`)
- **Classes**: 5 categories (Benign, DoS, Probe, R2L, U2R)

**Download**: 
- Official: [UNB NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)
- Alternative: [Kaggle NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd)

**Preprocessing**:
- Categorical encoding: protocol_type, service, flag
- Standardization: All numeric features (StandardScaler)
- Attack mapping: 38 attack types → 5 categories

---

## Known Issues & Limitations

1. **U2R Class Performance**: Low precision (13.51%) due to extreme class imbalance
   - **Mitigation**: Consider SMOTE or class weighting in future versions

2. **Window Length**: Currently fixed at 1 (no temporal context)
   - **Reason**: Aligns with training configuration
   - **Future**: Experiment with window_len > 1 for better temporal modeling

3. **Model Size**: 5.3 MB may exceed some free hosting limits
   - **Mitigation**: Model quantization or pruning

4. **Dataset Size**: Training/test files (~10 MB) not included in repo
   - **Mitigation**: Provide download script or links

---

## Performance Optimization

Applied optimizations:

- ✅ Removed all emoji/icons (professional code)
- ✅ Added comprehensive F1-score metrics (weighted, macro, per-class)
- ✅ Batch processing with progress tracking
- ✅ Cached model loading (`@st.cache_resource`)
- ✅ Type hints and docstrings for maintainability
- ✅ Centralized configuration via dataclass
- ✅ Production-grade logging

**Results**:
- Inference speed: 2,227 samples/sec
- Memory efficient: Batch processing prevents OOM
- UI responsive: Progress bars for long operations

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add type hints to new functions
- Write docstrings (Google style)
- Update tests for new features
- Ensure accuracy ≥ 80% on test set
- Update README if adding features

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{network-ids-dvrcnn,
  author = {Huy Nguyen },
  title = {Dual View CNN for Rare class Robust IDS},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/ngquochuy0101/network-ids-dvrcnn}
}
```

---

## Acknowledgments

- **NSL-KDD Dataset**: Canadian Institute for Cybersecurity, UNB
- **PyTorch**: Facebook AI Research
- **Streamlit**: Streamlit Inc.
- **Community**: Open-source ML/DL community

---

## Contact

- **Author**: [Huy Nguyen]
- **Email**: ngquochuy4002@gmail.com  
- **GitHub**: [@ngquochuy0101](https://github.com/ngquochuy0101   )

---

## Project Status

**Current Version**: 2.0.0 (MLOps Optimized)

**Roadmap**:
- [ ] Docker containerization
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Model quantization for faster inference
- [ ] Support for other IDS datasets (CICIDS2017, UNSW-NB15)
- [ ] Real-time network traffic monitoring
- [ ] REST API endpoint
- [ ] Model retraining pipeline

**Last Updated**: March 2026

---

**Built with ❤️ for Network Security**
