"""
Network Intrusion Detection System (IDS) - Production Version
==============================================================
MLOps-optimized Streamlit application for NSL-KDD dataset analysis
using DV-RCNN (Dual-View Residual CNN) model.

Author: MLOps Team
Version: 2.0.0 (Optimized)
"""

import os
import pickle
import logging
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import plotly.graph_objects as go

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class Config:
    """Application configuration"""
    # Device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    model_path: str = "model/best_model.pt"
    preprocessor_path: str = "plk/preprocessor.pkl"
    train_file: str = "dataset/KDDTrain+.txt"
    
    # Model parameters
    n_classes: int = 5
    window_len: int = 1
    image_h: int = 11
    batch_size: int = 128
    
    # Label mappings
    label_mapping: Dict[int, str] = None
    reverse_label_mapping: Dict[str, int] = None
    
    # Categorical columns for NSL-KDD
    categorical_cols: List[str] = None
    
    def __post_init__(self):
        if self.label_mapping is None:
            self.label_mapping = {
                0: 'Benign', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'
            }
        if self.reverse_label_mapping is None:
            self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        if self.categorical_cols is None:
            self.categorical_cols = ["protocol_type", "service", "flag"]

config = Config()

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class CNN1D(nn.Module):
    """1D Convolutional Neural Network with residual connections"""
    
    def __init__(self, in_channels: int, out_dim: int = 96):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 96, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(96)
        self.conv2 = nn.Conv1d(96, 192, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(192)
        self.conv3 = nn.Conv1d(192, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(384)
        
        self.shortcut = nn.Conv1d(in_channels, 384, kernel_size=1)
        self.bn_shortcut = nn.BatchNorm1d(384)
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(384, out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        h = F.relu(self.bn1(self.conv1(x)))
        if h.size(-1) > 1:
            h = F.max_pool1d(h, 2)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        
        identity = self.bn_shortcut(self.shortcut(identity))
        identity = F.adaptive_max_pool1d(identity, h.size(-1))
        
        h = h + identity
        h = F.adaptive_max_pool1d(h, 1).squeeze(-1)
        h = self.dropout(h)
        return self.fc(h)

class CNN2D(nn.Module):
    """2D Convolutional Neural Network for correlation matrices"""
    
    def __init__(self, out_dim: int = 96):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 48, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 96, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 192, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(192)
        self.conv4 = nn.Conv2d(192, 384, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(384)

        self.shortcut = nn.Conv2d(1, 384, kernel_size=1)
        self.bn_shortcut = nn.BatchNorm2d(384)

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(384, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        h = F.relu(self.bn1(self.conv1(x)))
        if h.size(-1) > 1:
            h = F.max_pool2d(h, 2)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))

        identity = self.bn_shortcut(self.shortcut(identity))
        identity = F.adaptive_max_pool2d(identity, h.shape[-2:])

        h = h + identity
        h = F.adaptive_max_pool2d(h, 1).squeeze(-1).squeeze(-1)
        h = self.dropout(h)
        return self.fc(h)

class AttentionFusion(nn.Module):
    """Attention-based fusion module for dual-view features"""
    
    def __init__(self, feature_dim: int = 96):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 2),
            nn.Softmax(dim=-1)
        )
        self.fc = nn.Linear(feature_dim * 2, feature_dim)
        
    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        concat_features = torch.cat([h1, h2], dim=-1)
        attention_weights = self.attention(concat_features)
        
        weighted_h1 = h1 * attention_weights[:, 0:1]
        weighted_h2 = h2 * attention_weights[:, 1:2]
        
        fused = torch.cat([weighted_h1, weighted_h2], dim=-1)
        return self.fc(fused)

class DVRCNN(nn.Module):
    """Dual-View Residual CNN for intrusion detection"""
    
    def __init__(self, d_in: int, n_classes: int):
        super().__init__()
        self.branch1d = CNN1D(in_channels=d_in, out_dim=96)
        self.branch2d = CNN2D(out_dim=96)
        self.fusion = AttentionFusion(feature_dim=96)
        
        self.classifier = nn.Sequential(
            nn.Linear(96, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(96, n_classes),
        )
        
    def forward(self, x1d: torch.Tensor, x2d: torch.Tensor) -> torch.Tensor:
        h1 = self.branch1d(x1d)
        h2 = self.branch2d(x2d)
        h_fused = self.fusion(h1, h2)
        return self.classifier(h_fused)

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
def load_nsl_kdd_file(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess NSL-KDD dataset file
    
    Args:
        file_path: Path to NSL-KDD .txt or .csv file
        
    Returns:
        DataFrame with processed data and attack_cat column
    """
    base_cols = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
        "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
        "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login",
        "is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
        "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
        "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
        "label","difficulty"
    ]
    
    try:
        df = pd.read_csv(file_path, header=None)
        
        if df.shape[1] in (42, 43):
            df.columns = base_cols[:df.shape[1]]
        
        df.drop(columns=["num_outbound_cmds"], inplace=True, errors='ignore')
        
        # Attack category mappings
        dos_attacks = {'apache2','mailbomb','back','land','neptune','pod','processtable',
                      'smurf','teardrop','udpstorm','worm'}
        probe_attacks = {'ipsweep','mscan','nmap','portsweep','saint','satan'}
        r2l_attacks = {'httptunnel','ftp_write','xlock','guess_passwd','http_tunnel','imap',
                       'multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy',
                       'warezclient','warezmaster','xclock','xsnoop'}
        u2r_attacks = {'buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm'}

        def map_attack_category(label: str) -> str:
            l = str(label).strip().lower()
            if l == "normal":
                return "Benign"
            elif l in dos_attacks:
                return "DoS"
            elif l in probe_attacks:
                return "Probe"
            elif l in r2l_attacks:
                return "R2L"
            elif l in u2r_attacks:
                return "U2R"
            return l

        df["attack_cat"] = df["label"].apply(map_attack_category)
        logger.info(f"Loaded {len(df)} records from {file_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        raise

def build_preprocessor(df: pd.DataFrame) -> Tuple:
    """
    Build preprocessing pipeline
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (preprocessor, X, y)
    """
    y_col = "attack_cat"
    y = df[y_col].astype(str)

    drop_cols = {c for c in df.columns 
                 if c.lower() in ("label","difficulty","binary","attack_cat","id","flow_id")}
    
    feat_cols = [c for c in df.columns if c not in drop_cols]
    num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feat_cols if c in config.categorical_cols]

    X = df[feat_cols].copy()

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=True, with_std=True), num_cols),
            ("cat", ohe, cat_cols),
        ]
    )
    
    logger.info(f"Preprocessor built: {len(num_cols)} numeric, {len(cat_cols)} categorical")
    return preprocessor, X, y

def get_or_create_preprocessor():
    """Load or create preprocessor from training data"""
    os.makedirs("plk", exist_ok=True)
    
    if os.path.exists(config.preprocessor_path):
        try:
            with open(config.preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            logger.info("Preprocessor loaded from cache")
            return preprocessor
        except Exception as e:
            logger.warning(f"Failed to load preprocessor: {e}. Creating new one...")
    
    # Create new preprocessor
    if not os.path.exists(config.train_file):
        raise FileNotFoundError(f"Training file not found: {config.train_file}")
    
    logger.info("Creating new preprocessor from training data...")
    train_df = load_nsl_kdd_file(config.train_file)
    preprocessor, X_train, _ = build_preprocessor(train_df)
    preprocessor.fit(X_train)
    
    # Save for future use
    with open(config.preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    logger.info("Preprocessor created and saved")
    return preprocessor

# ============================================================================
# WINDOW & IMAGE CREATION
# ============================================================================
def _resize_square(mat: np.ndarray, target: int) -> np.ndarray:
    """Resize square matrix to target size"""
    n = mat.shape[0]
    if n == target:
        return mat
    
    if n > target:
        if n % target == 0:
            factor = n // target
            return mat.reshape(target, factor, target, factor).mean(axis=(1,3))
        else:
            try:
                from skimage.transform import resize
                return resize(mat, (target, target), anti_aliasing=True)
            except ImportError:
                step = max(1, n // target)
                return mat[::step, ::step][:target, :target]
    else:
        factor = int(np.ceil(target / n))
        up = np.repeat(np.repeat(mat, factor, axis=0), factor, axis=1)
        start = (up.shape[0] - target) // 2
        return up[start:start+target, start:start+target]

def create_correlation_matrix_image(
    x_win: np.ndarray, 
    target_size: Tuple[int, int] = (11, 11)
) -> np.ndarray:
    """
    Create correlation matrix image from window
    
    Args:
        x_win: Input window array (L, D)
        target_size: Target image size (H, W)
        
    Returns:
        Correlation matrix resized to target_size
    """
    L, D = x_win.shape
    H, W = target_size
    
    eps = 1e-10
    feature_std = np.std(x_win, axis=0)
    constant_mask = feature_std < eps
    
    if np.all(constant_mask):
        return np.zeros((H, W), dtype=np.float32)
    
    if np.any(constant_mask):
        x_win = x_win.copy()
        x_win[:, constant_mask] += np.random.normal(scale=1e-8, size=(L, constant_mask.sum()))
    
    corr_matrix = np.corrcoef(x_win.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    np.fill_diagonal(corr_matrix, 1.0)
    
    corr_final = _resize_square(corr_matrix, H).astype(np.float32)
    return corr_final

def build_windows(
    X: np.ndarray, 
    window_len: int = 1, 
    stride: int = 1, 
    image_h: int = 11
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows for dual-view input
    
    Args:
        X: Input features (N, D)
        window_len: Window length
        stride: Stride for sliding window
        image_h: Target height for correlation matrix
        
    Returns:
        Tuple of (xs_1d, xs_2d) arrays
    """
    N, D = X.shape
    xs_1d, xs_2d = [], []
    H = image_h

    for start in range(0, N - window_len + 1, stride):
        end = start + window_len
        x_win = X[start:end]

        # 1D view: transpose to (D, L)
        x1d = x_win.T.astype(np.float32)

        # 2D view: correlation matrix
        x2d = create_correlation_matrix_image(x_win, target_size=(H, H))
        if x2d.ndim == 2:
            x2d = x2d[None, ...]

        xs_1d.append(x1d)
        xs_2d.append(x2d.astype(np.float32))

    xs_1d = np.stack(xs_1d, axis=0)
    xs_2d = np.stack(xs_2d, axis=0)

    logger.info(f"Created {xs_1d.shape[0]} windows")
    return xs_1d, xs_2d

# ============================================================================
# MODEL INFERENCE
# ============================================================================
@st.cache_resource
def load_model() -> Tuple[DVRCNN, int]:
    """
    Load pre-trained DV-RCNN model
    
    Returns:
        Tuple of (model, d_in)
    """
    if not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model file not found: {config.model_path}")
    
    try:
        checkpoint = torch.load(config.model_path, map_location=config.device, weights_only=False)
        
        # Determine d_in from checkpoint
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        first_conv_weight = state_dict['branch1d.conv1.weight']
        d_in = first_conv_weight.shape[1]
        
        # Create and load model
        model = DVRCNN(d_in=d_in, n_classes=config.n_classes)
        model.load_state_dict(state_dict)
        model.to(config.device)
        model.eval()
        
        logger.info(f"Model loaded successfully (d_in={d_in}, params={sum(p.numel() for p in model.parameters())})")
        return model, d_in
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def predict_batch(
    model: DVRCNN, 
    x1d_batch: np.ndarray, 
    x2d_batch: np.ndarray
) -> np.ndarray:
    """Predict single batch"""
    with torch.no_grad():
        x1d_t = torch.from_numpy(x1d_batch).to(config.device)
        x2d_t = torch.from_numpy(x2d_batch).to(config.device)
        logits = model(x1d_t, x2d_t)
        preds = logits.argmax(dim=-1).cpu().numpy()
    return preds

def predict_windows(
    model: DVRCNN, 
    xs_1d: np.ndarray, 
    xs_2d: np.ndarray, 
    batch_size: int = 128
) -> np.ndarray:
    """
    Predict all windows with batching
    
    Args:
        model: DV-RCNN model
        xs_1d: 1D view windows
        xs_2d: 2D view windows
        batch_size: Batch size for inference
        
    Returns:
        Predictions array
    """
    n_windows = xs_1d.shape[0]
    all_preds = []
    
    progress_bar = st.progress(0)
    for i in range(0, n_windows, batch_size):
        batch_x1d = xs_1d[i:i+batch_size]
        batch_x2d = xs_2d[i:i+batch_size]
        preds = predict_batch(model, batch_x1d, batch_x2d)
        all_preds.append(preds)
        progress_bar.progress(min((i + batch_size) / n_windows, 1.0))
    
    progress_bar.empty()
    logger.info(f"Prediction completed for {n_windows} windows")
    return np.concatenate(all_preds)

# ============================================================================
# METRICS CALCULATION
# ============================================================================
def calculate_comprehensive_metrics(
    y_true: List[str], 
    y_pred: List[str]
) -> Dict:
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing all metrics
    """
    # Convert to numeric
    y_true_numeric = [config.reverse_label_mapping[y] for y in y_true]
    y_pred_numeric = [config.reverse_label_mapping[y] for y in y_pred]
    
    # Overall metrics
    accuracy = accuracy_score(y_true_numeric, y_pred_numeric)
    
    # Weighted metrics
    precision_weighted = precision_score(y_true_numeric, y_pred_numeric, 
                                        average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true_numeric, y_pred_numeric, 
                                   average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true_numeric, y_pred_numeric, 
                          average='weighted', zero_division=0)
    
    # Macro metrics
    precision_macro = precision_score(y_true_numeric, y_pred_numeric, 
                                     average='macro', zero_division=0)
    recall_macro = recall_score(y_true_numeric, y_pred_numeric, 
                               average='macro', zero_division=0)
    f1_macro = f1_score(y_true_numeric, y_pred_numeric, 
                       average='macro', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true_numeric, y_pred_numeric, 
                                         average=None, zero_division=0)
    recall_per_class = recall_score(y_true_numeric, y_pred_numeric, 
                                    average=None, zero_division=0)
    f1_per_class = f1_score(y_true_numeric, y_pred_numeric, 
                           average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_numeric, y_pred_numeric)
    
    return {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm
    }

# ============================================================================
# STREAMLIT UI
# ============================================================================
def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="Hệ thống Phát hiện Xâm nhập Mạng",
        page_icon="shield",
        layout="wide"
    )
    
    st.title("Hệ thống Phát hiện Xâm nhập Mạng")
    st.markdown("### Mô hình DV-RCNN phân tích dữ liệu NSL-KDD")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Cấu hình Hệ thống")
        st.info(f"**Thiết bị:** {config.device}")
        st.info(f"**Mô hình:** DV-RCNN")
        st.info(f"**Tập dữ liệu:** NSL-KDD")
        st.info(f"**Số lớp:** {config.n_classes}")
        st.success(f"**Độ dài cửa sổ:** {config.window_len}")
        
        st.markdown("---")
        st.markdown("### Hướng dẫn Sử dụng")
        st.markdown("""
        1. Tải lên file dữ liệu kiểm tra NSL-KDD (định dạng .txt hoặc .csv)
        2. Xem trước dữ liệu đã tải
        3. Nhấn nút Phân tích để chạy dự đoán
        4. Xem kết quả và tải xuống các dự đoán
        """)
        
        st.markdown("---")
        st.markdown("### Các loại Tấn công")
        for label in config.label_mapping.values():
            st.text(f"- {label}")
    
    # Load model
    try:
        with st.spinner("Đang tải mô hình..."):
            model, d_in = load_model()
        st.success(f"Đã tải mô hình thành công (số chiều đầu vào: {d_in})")
    except Exception as e:
        st.error(f"Không thể tải mô hình: {str(e)}")
        logger.error(f"Model loading error: {e}", exc_info=True)
        st.stop()
    
    # Load preprocessor
    try:
        with st.spinner("Đang tải bộ tiền xử lý..."):
            preprocessor = get_or_create_preprocessor()
        st.success("Bộ tiền xử lý đã sẵn sàng")
    except Exception as e:
        st.error(f"Không thể tải bộ tiền xử lý: {str(e)}")
        logger.error(f"Preprocessor loading error: {e}", exc_info=True)
        st.stop()
    
    # File upload
    st.header("Tải lên Dữ liệu")
    uploaded_file = st.file_uploader(
        "Chọn file dữ liệu kiểm tra NSL-KDD",
        type=['txt', 'csv'],
        help="Tải lên file KDDTest+.txt hoặc định dạng tương tự"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            with st.spinner("Đang tải dữ liệu..."):
                df = load_nsl_kdd_file(uploaded_file)
            
            st.success(f"Đã tải {len(df)} bản ghi")
            
            # Preview
            st.header("Xem trước Dữ liệu")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tổng số Bản ghi", len(df))
            with col2:
                st.metric("Số Đặc trưng", len(df.columns) - 3)
            with col3:
                if 'attack_cat' in df.columns:
                    n_attacks = (df['attack_cat'] != 'Benign').sum()
                    st.metric("Số Tấn công (Thực tế)", n_attacks)
            
            # Analyze button
            if st.button("Phân tích", type="primary", use_container_width=True):
                with st.spinner("Đang phân tích..."):
                    
                    # Preprocessing
                    st.info("Bước 1/4: Đang tiền xử lý dữ liệu...")
                    _, X, _ = build_preprocessor(df)
                    X_transformed = preprocessor.transform(X)
                    
                    # Build windows
                    st.info("Bước 2/4: Đang tạo cửa sổ dữ liệu...")
                    xs_1d, xs_2d = build_windows(
                        X_transformed,
                        window_len=config.window_len,
                        stride=1,
                        image_h=config.image_h
                    )
                    st.info(f"Đã tạo {xs_1d.shape[0]} cửa sổ")
                    
                    # Prediction
                    st.info("Bước 3/4: Đang chạy dự đoán...")
                    window_preds = predict_windows(model, xs_1d, xs_2d, 
                                                  batch_size=config.batch_size)
                    
                    # Map to samples (1:1 for window_len=1)
                    st.info("Bước 4/4: Đang xử lý kết quả...")
                    sample_preds = window_preds
                    
                    pred_labels = [config.label_mapping[p] for p in sample_preds]
                    df['prediction'] = pred_labels
                    
                st.success("Phân tích hoàn tất")
                
                # ============================================================
                # RESULTS DISPLAY
                # ============================================================
                st.header("Kết quả Phân tích")
                
                # Basic metrics
                n_benign = (df['prediction'] == 'Benign').sum()
                n_attack = len(df) - n_benign
                attack_rate = n_attack / len(df) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Tổng số Mẫu", len(df))
                with col2:
                    st.metric("Bình thường", n_benign)
                with col3:
                    st.metric("Tấn công", n_attack)
                with col4:
                    st.metric("Tỷ lệ Tấn công", f"{attack_rate:.2f}%")
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Phân bố Bình thường và Tấn công")
                    category_counts = df['prediction'].apply(
                        lambda x: 'Bình thường' if x == 'Benign' else 'Tấn công'
                    ).value_counts()
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=category_counts.index,
                        values=category_counts.values,
                        hole=0.3,
                        marker_colors=['#00cc96', '#ef553b']
                    )])
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Phân bố theo Loại Tấn công")
                    attack_counts = df['prediction'].value_counts()
                    
                    fig = go.Figure(data=[go.Bar(
                        x=attack_counts.index,
                        y=attack_counts.values,
                        marker_color=['#00cc96' if x == 'Benign' else '#ef553b' 
                                     for x in attack_counts.index]
                    )])
                    fig.update_layout(
                        xaxis_title="Loại Tấn công",
                        yaxis_title="Số lượng",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                st.subheader("Kết quả Chi tiết")
                display_cols = ['duration', 'protocol_type', 'service', 'flag', 
                               'src_bytes', 'dst_bytes', 'prediction']
                available_cols = [c for c in display_cols if c in df.columns]
                st.dataframe(df[available_cols].head(20), use_container_width=True)
                
                # Download button
                st.subheader("Tải xuống Kết quả")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Tải xuống CSV",
                    data=csv,
                    file_name="ket_qua_phan_tich.csv",
                    mime="text/csv",
                    type="primary"
                )
                
                # ============================================================
                # GROUND TRUTH COMPARISON (if available)
                # ============================================================
                if 'attack_cat' in df.columns:
                    st.subheader("So sánh với Nhãn Thực tế")
                    
                    true_labels = df['attack_cat'].tolist()
                    pred_labels_list = df['prediction'].tolist()
                    
                    # Calculate comprehensive metrics
                    metrics = calculate_comprehensive_metrics(true_labels, pred_labels_list)
                    
                    # Overall metrics display
                    st.markdown("#### Hiệu suất Tổng thể")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Độ chính xác", f"{metrics['accuracy']*100:.2f}%")
                    with col2:
                        st.metric("Precision (Trọng số)", f"{metrics['precision_weighted']*100:.2f}%")
                    with col3:
                        st.metric("Recall (Trọng số)", f"{metrics['recall_weighted']*100:.2f}%")
                    with col4:
                        st.metric("F1-Score (Trọng số)", f"{metrics['f1_weighted']*100:.2f}%")
                    
                
                    # Per-class metrics table
                    st.markdown("#### Hiệu suất theo từng Lớp")
                    class_metrics_data = []
                    for i, class_name in enumerate(config.label_mapping.values()):
                        support = (df['attack_cat'] == class_name).sum()
                        class_metrics_data.append({
                            "Lớp": class_name,
                            "Precision": f"{metrics['precision_per_class'][i]*100:.2f}%",
                            "Recall": f"{metrics['recall_per_class'][i]*100:.2f}%",
                            "F1-Score": f"{metrics['f1_per_class'][i]*100:.2f}%",
                            "Số mẫu": support
                        })
                    
                    metrics_df = pd.DataFrame(class_metrics_data)
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                    
                    # Confusion matrix
                    st.markdown("#### Ma trận Nhầm lẫn")
                    cm = metrics['confusion_matrix']
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=list(config.label_mapping.values()),
                        y=list(config.label_mapping.values()),
                        colorscale='Blues',
                        text=cm,
                        texttemplate="%{text}",
                        textfont={"size": 12}
                    ))
                    fig.update_layout(
                        xaxis_title="Dự đoán",
                        yaxis_title="Thực tế",
                        yaxis=dict(autorange='reversed'),
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # # Performance summary
                    # if metrics['accuracy'] >= 0.80:
                    #     st.success(f"""
                    #     **Excellent Performance Achieved**
                    #     - Overall Accuracy: {metrics['accuracy']*100:.2f}%
                    #     - Weighted F1-Score: {metrics['f1_weighted']*100:.2f}%
                    #     - Model performing within expected range (80%+)
                    #     """)
                    # else:
                    #     st.info(f"""
                    #     **Performance Summary**
                    #     - Overall Accuracy: {metrics['accuracy']*100:.2f}%
                    #     - Weighted F1-Score: {metrics['f1_weighted']*100:.2f}%
                    #     """)
        
        except Exception as e:
            st.error(f"Lỗi khi xử lý file: {str(e)}")
            logger.error(f"Processing error: {e}", exc_info=True)
            st.exception(e)
    
    else:
        st.info("Vui lòng tải lên file để bắt đầu phân tích")
        
        # Info section
        st.markdown("---")
        st.header("Thông tin về Mô hình")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Kiến trúc DV-RCNN
            - **Mạng CNN hai luồng** gồm hai nhánh:
              - CNN1D: Xử lý đặc trưng tuần tự
              - CNN2D: Phân tích ma trận tương quan
            - **Kết hợp Attention**: Tổng hợp đặc trưng thông minh
            - **Kết nối Dư**: Cải thiện luồng gradient
            """)
        
        with col2:
            st.markdown("""
            ### Chi tiết Kỹ thuật
            - **Độ dài cửa sổ**: 1 (khớp với cấu hình huấn luyện)
            - **Kích thước batch**: 128 mẫu
            - **Kích thước ảnh**: Ma trận tương quan 11x11
            - **Các chỉ số**: Độ chính xác, Precision, Recall, F1-Score
            """)

if __name__ == "__main__":
    main()
