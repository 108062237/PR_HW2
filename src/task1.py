import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from data_loader import load_config, load_dataset # 假設您的 data_loader.py 在同一個目錄或PYTHONPATH中
import argparse
import os # 用於從路徑中獲取檔案名稱

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/breast_cancer.yaml', help='Path to YAML config file')
    args = parser.parse_args()
    return args

def calculate_scatter_matrices(X, y):
    """
    計算類內散佈矩陣 S_W 和類間散佈矩陣 S_B。
    """
    n_features = X.shape[1]
    classes = np.unique(y)
    n_classes = len(classes)

    S_W = np.zeros((n_features, n_features))
    class_means = {} 

    for c_val in classes:
        X_c = X[y == c_val]
        class_means[c_val] = np.mean(X_c, axis=0)
        
        s_i = np.zeros((n_features, n_features))
        m_c_reshaped = class_means[c_val].reshape(n_features, 1)
        for row_idx in range(X_c.shape[0]):
            x_sample = X_c[row_idx, :].reshape(n_features, 1)
            s_i += (x_sample - m_c_reshaped).dot((x_sample - m_c_reshaped).T)
        S_W += s_i

    m_global = np.mean(X, axis=0)
    S_B = np.zeros((n_features, n_features))
    m_global_reshaped = m_global.reshape(n_features, 1)
    for c_val in classes:
        N_c = X[y == c_val].shape[0]
        m_c_reshaped = class_means[c_val].reshape(n_features, 1)
        S_B += N_c * (m_c_reshaped - m_global_reshaped).dot((m_c_reshaped - m_global_reshaped).T)

    return S_W, S_B

def print_separability_measures(S_W, S_B, description=""):
    """
    印出基於 S_W 和 S_B 的可分性度量。
    """
    trace_S_W = np.trace(S_W)
    trace_S_B = np.trace(S_B)
    
    print(f"\n--- Separability Measures: {description} ---")
    print(f"Trace(S_W): {trace_S_W:.4f} ")
    print(f"Trace(S_B): {trace_S_B:.4f} ")
    
    if trace_S_W != 0:
        ratio_S_B_S_W = trace_S_B / trace_S_W
        print(f"Trace(S_B) / Trace(S_W): {ratio_S_B_S_W:.4f} ")
    else:
        print("Trace(S_B) / Trace(S_W): Undefined (Trace(S_W) is zero)")
    print("-----------------------------------------")


if __name__ == "__main__":
    args = get_args()
    config_path = args.config
    
    # 從 config_path 提取資料集名稱用於標題
    try:
        dataset_name = os.path.splitext(os.path.basename(config_path))[0]
    except Exception:
        dataset_name = "Unknown Dataset"

    print(f"Loading dataset from: {config_path} (Dataset: {dataset_name})")
    cfg = load_config(config_path)
    X_train, X_test, y_train, y_test = load_dataset(cfg)

    # 確保 y 是 numpy array，方便後續處理
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # 1. 特徵標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"\nOriginal number of features: {X_train_scaled.shape[1]}")

    # 2. 計算投影前的可分性度量 (使用標準化後的訓練數據)
    S_W_before_train, S_B_before_train = calculate_scatter_matrices(X_train_scaled, y_train)
    S_B_before_test, S_W_before_test = calculate_scatter_matrices(X_test_scaled, y_test)
    print_separability_measures(S_W_before_train, S_B_before_train, "Before FLD/LDA Projection (on Training Data)")
    print_separability_measures(S_W_before_test, S_B_before_test, "Before FLD/LDA Projection (on Test Data)")

    # 3. 應用 FLD/LDA
    # 對於兩類別問題，n_components 最多為 1
    # 對於 C 個類別問題，n_components 最多為 C-1
    # n_components=None 會自動選擇 min(n_classes - 1, n_features)
    n_classes = len(np.unique(y_train))
    # 確保 n_components 不會是 0 (當 n_classes=1)
    lda_n_components = None
    if n_classes == 1:
        print("Warning: Only one class found in training data. LDA will not perform dimensionality reduction meaningfully.")
        # 在這種情況下，LDA 可能會出錯或沒有意義，可以考慮不降維或只用一個 component (如果特徵允許)
        # 或者直接跳過LDA部分，因為可分性沒有意義
        lda_n_components = min(1, X_train_scaled.shape[1]) # 至少有一個component，如果特徵數允許
    
    lda = LDA(n_components=lda_n_components) # 使用 lda_n_components

    # 只有在類別數 > 1 時才 fit 和 transform
    if n_classes > 1:
        lda.fit(X_train_scaled, y_train)
        X_train_lda = lda.transform(X_train_scaled)
        X_test_lda = lda.transform(X_test_scaled)
        print(f"Reduced number of features (FLD/LDA): {X_train_lda.shape[1]}")

        # 4. 計算投影後的可分性度量 (使用投影後的訓練數據)
        S_W_after, S_B_after = calculate_scatter_matrices(X_train_lda, y_train)
        S_W_after_test, S_B_after_test = calculate_scatter_matrices(X_test_lda, y_test)
        print_separability_measures(S_W_after, S_B_after, "After FLD/LDA Projection (on Training Data)")
        print_separability_measures(S_W_after_test, S_B_after_test, "After FLD/LDA Projection (on Test Data)")


        if n_classes > 1 and X_train_lda.shape[1] == 2: # 只有當投影到2D時才方便繪製
            plt.figure(figsize=(8, 6))
            for i, target_name in enumerate(np.unique(y_train)): # 假設 y_train 中的類別是 0, 1, 2...
                                                              # 如果您的類別標籤不是這樣，可能需要映射到名稱
                # 您可能需要一個從類別索引到類別名稱的映射，如果 cfg['target_names'] 或類似的可用
                # 為了通用性，我們先用類別索引作為標籤
                plt.scatter(X_train_lda[y_train == target_name, 0], 
                            X_train_lda[y_train == target_name, 1], 
                            label=f'Class {target_name}')
            plt.xlabel('LD1 (Linear Discriminant 1)')
            plt.ylabel('LD2 (Linear Discriminant 2)')
            plt.title(f'FLD/LDA Projection of {dataset_name} (Training Data)')
            plt.legend()
            plt.grid(True)
            lda_scatter_filename = f"fld_lda_scatter_{dataset_name.lower().replace(' ', '_')}.png"
            plt.savefig(lda_scatter_filename)
            print(f"LDA scatter plot saved to {lda_scatter_filename}")
            plt.show()

        # 5. 對於兩類別資料集，繪製ROC曲線及計算AUC (使用測試數據)
        if n_classes == 2:
            print("\nTwo-class dataset detected. Calculating ROC and AUC for test set...")
            # 使用 lda 的 predict_proba 獲取正類別的機率分數
            # lda.classes_ 屬性會告訴您類別的順序，通常 predict_proba 的第二列對應 classes_[1]
            try:
                y_scores = lda.predict_proba(X_test_scaled)[:, 1]
                
                fpr, tpr, thresholds = roc_curve(y_test, y_scores)
                roc_auc = auc(fpr, tpr)

                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {dataset_name} (FLD/LDA)')
                plt.legend(loc="lower right")
                plt.grid(True)
                # 儲存圖片
                plot_filename = f"roc_fld_lda_{dataset_name.lower().replace(' ', '_')}.png"
                plt.savefig(plot_filename)
                print(f"ROC curve saved to {plot_filename}")
                plt.show()

            except Exception as e:
                print(f"Error during ROC/AUC calculation or plotting: {e}")
                print("This might happen if the LDA model has issues or only one class is predicted.")
        else:
            print(f"\nDataset has {n_classes} classes. ROC/AUC plot is typically for two-class problems.")

    else: # n_classes <= 1
        print("\nFLD/LDA and further analysis skipped due to insufficient number of classes.")

    print("\n--- Script Finished ---")