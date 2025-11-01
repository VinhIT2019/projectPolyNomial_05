import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import (
    LinearRegression,
    Ridge, RidgeCV,
    Lasso, LassoCV,
    ElasticNet, ElasticNetCV
)
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline



# Tái lặp lại kết quả

SEED = 42
np.random.seed(SEED)



# Các hàm tiện ích

def make_poly(degree):
    return PolynomialFeatures(degree=degree, include_bias=False)


def build_tuning_pipeline(
    model_name: str,
    degree_poly: int,
    ridge_alpha_min_exp: float,
    ridge_alpha_max_exp: float,
    ridge_alpha_num: int,
    lasso_n_alphas: int,
    lasso_max_iter: int,
    enet_l1_list,
    enet_alpha_min_exp: float,
    enet_alpha_max_exp: float,
    enet_alpha_num: int
):
    """
    Pipeline "tuning" dùng để tìm siêu tham số tốt nhất.
    Dựa trên model_name:
    - LinearRegression (no polynomial): scaler -> LinearRegression
    - Poly3 + LinearRegression: scaler -> poly3 -> LinearRegression
    - Poly4 + RidgeCV: poly4 -> scaler -> RidgeCV
    - Poly4 + LassoCV: poly4 -> scaler -> LassoCV
    - Poly4 + ElasticNetCV: poly4 -> scaler -> ElasticNetCV
    """
    if model_name == "LinearRegression (no polynomial)":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])
        return pipe

    if model_name == "Poly3 + LinearRegression":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("poly_features", make_poly(3)),
            ("model", LinearRegression())
        ])
        return pipe

    if model_name == "Poly4 + RidgeCV":
        alphas_ridge = np.logspace(
            ridge_alpha_min_exp,
            ridge_alpha_max_exp,
            ridge_alpha_num
        )
        pipe = Pipeline([
            ("poly_features", make_poly(degree_poly)),
            ("scaler_poly", StandardScaler()),
            ("model", RidgeCV(alphas=alphas_ridge))
        ])
        return pipe

    if model_name == "Poly4 + LassoCV":
        pipe = Pipeline([
            ("poly_features", make_poly(degree_poly)),
            ("scaler_poly", StandardScaler()),
            ("model", LassoCV(
                n_alphas=lasso_n_alphas,
                cv=5,
                max_iter=lasso_max_iter,
                random_state=SEED
            ))
        ])
        return pipe

    if model_name == "Poly4 + ElasticNetCV":
        alphas_enet = np.logspace(
            enet_alpha_min_exp,
            enet_alpha_max_exp,
            enet_alpha_num
        )
        pipe = Pipeline([
            ("poly_features", make_poly(degree_poly)),
            ("scaler_poly", StandardScaler()),
            ("model", ElasticNetCV(
                l1_ratio=list(enet_l1_list),
                alphas=alphas_enet,
                cv=5,
                max_iter=200000,
                tol=1e-3,
                random_state=SEED
            ))
        ])
        return pipe

    raise ValueError("Model chưa được hỗ trợ.")


def build_final_frozen_pipeline(
    model_name: str,
    tuning_pipeline: Pipeline,
    X_train,
    y_train,
    degree_poly: int,
    lasso_max_iter: int,
    enet_tol: float = 1e-3
):

    if model_name in ["LinearRegression (no polynomial)", "Poly3 + LinearRegression"]:
        # không có hyperparam để chọn
        return tuning_pipeline, {}

    # Fit tuning pipeline
    tuning_pipeline.fit(X_train, y_train)
    model_step = tuning_pipeline.named_steps["model"]

    if model_name == "Poly4 + RidgeCV":
        best_alpha = model_step.alpha_
        final_pipeline = Pipeline([
            ("poly_features", make_poly(degree_poly)),
            ("scaler_poly", StandardScaler()),
            ("model", Ridge(alpha=best_alpha, random_state=SEED))
        ])
        best_params = {"alpha": best_alpha}
        return final_pipeline, best_params

    if model_name == "Poly4 + LassoCV":
        best_alpha = model_step.alpha_
        final_pipeline = Pipeline([
            ("poly_features", make_poly(degree_poly)),
            ("scaler_poly", StandardScaler()),
            ("model", Lasso(alpha=best_alpha,
                            max_iter=lasso_max_iter,
                            random_state=SEED))
        ])
        best_params = {"alpha": best_alpha}
        return final_pipeline, best_params

    if model_name == "Poly4 + ElasticNetCV":
        best_alpha = model_step.alpha_
        best_l1_ratio = model_step.l1_ratio_
        final_pipeline = Pipeline([
            ("poly_features", make_poly(degree_poly)),
            ("scaler_poly", StandardScaler()),
            ("model", ElasticNet(
                alpha=best_alpha,
                l1_ratio=best_l1_ratio,
                max_iter=200000,
                tol=enet_tol,
                random_state=SEED
            ))
        ])
        best_params = {
            "alpha": best_alpha,
            "l1_ratio": best_l1_ratio
        }
        return final_pipeline, best_params

    raise ValueError("Model chưa được hỗ trợ trong bước freeze.")


def evaluate_holdout(model, X_train, X_test, y_train, y_test):
    """
    Fit model trên tập train (full pipeline),
    dự đoán test, tính R² train / R² test / RMSE test.
    """
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)

    rmse_test = sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)

    return {
        "R2_train": r2_train,
        "R2_test": r2_test,
        "RMSE_test": rmse_test
    }, y_pred_test


def compute_cv_scores_final(model, X, y, seed=SEED):
    
    cv_strategy = RepeatedKFold(
        n_splits=10,
        n_repeats=3,
        random_state=seed
    )

    scores_r2 = cross_val_score(
        model,
        X, y,
        cv=cv_strategy,
        scoring='r2'
    )
    scores_rmse = cross_val_score(
        model,
        X, y,
        cv=cv_strategy,
        scoring='neg_root_mean_squared_error'
    )

    return {
        "R2_CV_mean": np.mean(scores_r2),
        "R2_CV_std": np.std(scores_r2),
        "RMSE_CV_mean": -np.mean(scores_rmse),
        "RMSE_CV_std": np.std(scores_rmse),
    }


def extract_coefficients(trained_pipeline, feature_names_original):
    """
    Trả về top hệ số dương / âm nếu model cuối có .coef_.
    """
    poly_step_name = None
    for name, step in trained_pipeline.named_steps.items():
        if isinstance(step, PolynomialFeatures):
            poly_step_name = name
            break

    # bước model cuối
    model_step_name = list(trained_pipeline.named_steps.keys())[-1]
    model_step = trained_pipeline.named_steps[model_step_name]

    if not hasattr(model_step, "coef_"):
        return None, None

    coefs = model_step.coef_
    if len(np.shape(coefs)) > 1:
        return None, None

    if poly_step_name is None:
        used_feature_names = feature_names_original
    else:
        poly_step = trained_pipeline.named_steps[poly_step_name]
        used_feature_names = poly_step.get_feature_names_out(feature_names_original)

    coef_df = pd.DataFrame({
        "Feature": used_feature_names,
        "Coefficient": coefs
    })

    coef_sorted_desc = coef_df.sort_values(by="Coefficient", ascending=False)
    coef_sorted_asc = coef_df.sort_values(by="Coefficient", ascending=True)

    return coef_sorted_desc, coef_sorted_asc


def scatter_true_pred(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], linestyle='--')
    ax.set_xlabel("Giá trị thật (y_test)")
    ax.set_ylabel("Dự đoán (y_pred)")
    ax.set_title("So sánh Dự đoán vs Thực tế")
    return fig



# Giao diện Streamlit


st.set_page_config(
    page_title="Project: Concrete Compressive Strength (CONQ029 - AIO2025)",
    layout="wide"
)

st.title("Project: Concrete Compressive Strength (CONQ029 - AIO2025)")
st.markdown(
    """
    > ##### Project nhằm dự đoán sức mạnh nén của bê tông (Concrete Compressive Strength) dựa trên các thành phần vật liệu được pha trộn.
    """
)

# ===== SIDEBAR: CẤU HÌNH =====
st.sidebar.header("1. Dữ liệu đầu vào")

uploaded_file = st.sidebar.file_uploader(
    "Tải dataset (.csv, .xlsx)",
    type=["csv", "xlsx"]
)

test_size = st.sidebar.slider(
    "Tỉ lệ test_size",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05
)

random_state = st.sidebar.number_input(
    "random_state (tái lặp thí nghiệm)",
    min_value=0,
    value=SEED,
    step=1
)

st.sidebar.header("2. Mô hình")

model_name = st.sidebar.selectbox(
    "Pipeline huấn luyện",
    [
        "LinearRegression (no polynomial)",
        "Poly3 + LinearRegression",
        "Poly4 + RidgeCV",
        "Poly4 + LassoCV",
        "Poly4 + ElasticNetCV"
    ]
)

st.sidebar.header("3. Tham số mô hình")

# Các tham số mặc định
ridge_alpha_min_exp = -7.0
ridge_alpha_max_exp = 10.0
ridge_alpha_num = 200

lasso_n_alphas = 100
lasso_max_iter = 200000

enet_alpha_min_exp = -3.0
enet_alpha_max_exp = 1.0
enet_alpha_num = 10
enet_l1_low = 0.3
enet_l1_mid = 0.5
enet_l1_high = 0.7

if model_name == "Poly4 + RidgeCV":
    st.sidebar.subheader("RidgeCV hyperparams")
    ridge_alpha_min_exp = st.sidebar.number_input(
        "min exp alpha (10^min)",
        value=-7.0, step=1.0
    )
    ridge_alpha_max_exp = st.sidebar.number_input(
        "max exp alpha (10^max)",
        value=10.0, step=1.0
    )
    ridge_alpha_num = st.sidebar.number_input(
        "Số lượng alpha trong logspace",
        min_value=10, max_value=2000, value=200, step=10
    )

elif model_name == "Poly4 + LassoCV":
    st.sidebar.subheader("LassoCV hyperparams")
    lasso_n_alphas = st.sidebar.number_input(
        "n_alphas (LassoCV)",
        min_value=10, max_value=1000,
        value=100, step=10
    )
    lasso_max_iter = st.sidebar.number_input(
        "max_iter (Lasso/LassoCV)",
        min_value=10000, max_value=500000,
        value=200000, step=10000
    )

elif model_name == "Poly4 + ElasticNetCV":
    st.sidebar.subheader("ElasticNetCV hyperparams")
    enet_alpha_min_exp = st.sidebar.number_input(
        "min exp alpha (10^min)",
        value=-3.0, step=1.0
    )
    enet_alpha_max_exp = st.sidebar.number_input(
        "max exp alpha (10^max)",
        value=1.0, step=1.0
    )
    enet_alpha_num = st.sidebar.number_input(
        "Số alpha",
        min_value=5, max_value=200,
        value=10, step=1
    )
    enet_l1_low = st.sidebar.slider(
        "l1_ratio thấp",
        min_value=0.1, max_value=0.9,
        value=0.3, step=0.1
    )
    enet_l1_mid = st.sidebar.slider(
        "l1_ratio giữa",
        min_value=0.1, max_value=0.9,
        value=0.5, step=0.1
    )
    enet_l1_high = st.sidebar.slider(
        "l1_ratio cao",
        min_value=0.1, max_value=0.9,
        value=0.7, step=0.1
    )

# chọn bậc đa thức
if model_name == "Poly3 + LinearRegression":
    degree_poly = 3
elif model_name in ["Poly4 + RidgeCV", "Poly4 + LassoCV", "Poly4 + ElasticNetCV"]:
    degree_poly = 4
else:
    degree_poly = 1  # không dùng cho linear no-poly

train_button = st.sidebar.button("Train / Evaluate model")


# ===== MAIN LAYOUT =====
col_data, col_results = st.columns([1, 1.2])

with col_data:
    st.subheader("Bảng dữ liệu")

    if uploaded_file is not None:
        # Đọc file
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Loại duplicate
        before_n = df.shape[0]
        df = df.drop_duplicates().reset_index(drop=True)
        after_n = df.shape[0]

        st.write("Kích thước dữ liệu sau khi loại duplicate:", df.shape)
        if after_n < before_n:
            st.info(f"Đã loại {before_n - after_n} dòng duplicate.")

        st.dataframe(df.head(5))

        # Chọn cột target mặc định = cột cuối
        all_cols = list(df.columns)
        default_target = all_cols[-1] if len(all_cols) > 0 else None

        target_col = st.selectbox(
            "Chọn cột target (biến cần dự đoán)",
            options=all_cols,
            index=(all_cols.index(default_target) if default_target in all_cols else 0)
        )

        default_features = [c for c in all_cols if c != target_col]

        feature_cols = st.multiselect(
            "Chọn các feature đầu vào (X)",
            options=[c for c in all_cols if c != target_col],
            default=default_features
        )

    else:
        df = None
        target_col = None
        feature_cols = None
        st.info("Hãy upload dataset ở sidebar để tiếp tục.")


with col_results:
    st.subheader("Kết quả huấn luyện / đánh giá")

    if train_button and df is not None and target_col is not None and len(feature_cols) > 0:
        # Chuẩn bị dữ liệu
        X = df[feature_cols].values
        y = df[target_col].values

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        # 1. Pipeline tuning
        tuning_pipeline = build_tuning_pipeline(
            model_name=model_name,
            degree_poly=degree_poly,
            ridge_alpha_min_exp=ridge_alpha_min_exp,
            ridge_alpha_max_exp=ridge_alpha_max_exp,
            ridge_alpha_num=ridge_alpha_num,
            lasso_n_alphas=lasso_n_alphas,
            lasso_max_iter=lasso_max_iter,
            enet_l1_list=(enet_l1_low, enet_l1_mid, enet_l1_high),
            enet_alpha_min_exp=enet_alpha_min_exp,
            enet_alpha_max_exp=enet_alpha_max_exp,
            enet_alpha_num=enet_alpha_num
        )

        # 2. Freeze hyperparams để tạo final pipeline
        final_pipeline, best_params = build_final_frozen_pipeline(
            model_name=model_name,
            tuning_pipeline=tuning_pipeline,
            X_train=X_train,
            y_train=y_train,
            degree_poly=degree_poly,
            lasso_max_iter=lasso_max_iter
        )

        # 3. Đánh giá hold-out
        holdout_metrics, y_pred_test = evaluate_holdout(
            final_pipeline,
            X_train, X_test,
            y_train, y_test
        )

        # 4. Đánh giá CV RepeatedKFold(10×3) trên final pipeline
        cv_scores = compute_cv_scores_final(
            final_pipeline,
            X, y,
            seed=random_state
        )


        st.markdown("##### Hold-out metrics (train/test split)")
        
        st.markdown(f"> ##### R² train: {holdout_metrics['R2_train']:.4f} || R² test: {holdout_metrics['R2_test']:.4f} || RMSE test: {holdout_metrics['RMSE_test']:.4f}")
        
        st.markdown("##### Cross-Validation (RepeatedKFold 10-fold × 3 repeats)")
        st.markdown(f" > ##### R² CV mean = {cv_scores['R2_CV_mean']:.4f} ± {cv_scores['R2_CV_std']:.4f} || RMSE CV mean = {cv_scores['RMSE_CV_mean']:.4f} ± {cv_scores['RMSE_CV_std']:.4f}")
              
        # Scatter plot
        st.markdown("#### So sánh Dự đoán vs Thực tế (Test set)")
        fig_scatter = scatter_true_pred(y_test, y_pred_test)
        st.pyplot(fig_scatter)

        # Hệ số tuyến tính
        st.markdown("#### Phân tích hệ số (Feature Importance tuyến tính)")
        coef_pos, coef_neg = extract_coefficients(final_pipeline, feature_cols)
        if coef_pos is None:
            st.write("Không trích xuất được hệ số (mô hình không có coef_).")
        else:
            st.write("Top 10 đặc trưng ảnh hưởng TÍCH CỰC nhất:")
            st.dataframe(coef_pos.head(10))

            st.write("Top 10 đặc trưng ảnh hưởng TIÊU CỰC nhất:")
            st.dataframe(coef_neg.head(10))

    elif train_button:
        st.warning("Thiếu dữ liệu / target / feature để train.")


