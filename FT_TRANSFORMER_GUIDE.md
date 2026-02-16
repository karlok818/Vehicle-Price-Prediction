# FT-Transformer Quick Integration Guide

## 📋 Overview

This guide helps you add FT-Transformer to your vehicle price regression notebook. FT-Transformer is a deep learning model that:
- ✅ **No scaling required** - Works with raw numerical features
- ✅ **No encoding required** - Handles categorical features natively
- ✅ **State-of-the-art** - Transformer architecture for tabular data
- ✅ **GPU support** - Automatically uses GPU if available

## 🔧 Installation

Run these commands in a notebook cell or terminal:

```python
# Install PyTorch (CPU version)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install RTDL (Research on Tabular Deep Learning)
!pip install rtdl
```

**For GPU support** (if you have CUDA installed):
```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install rtdl
```

## 📝 Integration Steps

### Option 1: Copy Individual Cells (Recommended)

Open `ft_transformer_cells.md` and copy each cell section into your notebook after Section 4.

### Option 2: Use the Python Class

1. Import the implementation:
```python
from ft_transformer_implementation import FTTransformerPreprocessor, FTTransformerTrainer
```

2. Use in your notebook:

```python
# 1. Preprocess data
preprocessor = FTTransformerPreprocessor()
train_ft = preprocessor.preprocess(df_train, is_train=True)
test_ft = preprocessor.preprocess(df_test, is_train=False)

# 2. Prepare features
X_train_ft = train_ft.drop(columns=['Sold_Amount'])
y_train_ft = train_ft['Sold_Amount']
X_test_ft = test_ft.drop(columns=['Sold_Amount'], errors='ignore')

# 3. Encode categorical features (simple label encoding)
X_train_encoded, X_test_encoded = preprocessor.encode_features(X_train_ft, X_test_ft)

# 4. Train-test split
from sklearn.model_selection import train_test_split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_encoded, np.log1p(y_train_ft), test_size=0.2, random_state=42
)

# 5. Get cardinalities
cat_cardinalities = [int(X_train_split[col].max() + 1) for col in preprocessor.cat_features]
n_num_features = len(preprocessor.num_features)

# 6. Create trainer and build model
trainer = FTTransformerTrainer(device='cuda')  # or 'cpu'
trainer.build_model(n_num_features, cat_cardinalities)

# 7. Prepare data loaders
train_loader, val_loader = trainer.prepare_data_loaders(
    X_train_split, y_train_split.values,
    X_val_split, y_val_split.values,
    preprocessor.cat_features,
    preprocessor.num_features,
    batch_size=256
)

# 8. Train
train_losses, val_losses = trainer.train(
    train_loader, val_loader,
    n_epochs=50, lr=0.001, patience=10
)

# 9. Evaluate
y_pred_log, y_actual_log = trainer.evaluate(val_loader)
y_pred = np.expm1(y_pred_log)
y_actual = np.expm1(y_actual_log)

# 10. Calculate metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae = mean_absolute_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)

print(f"RMSE: ${rmse:,.0f}")
print(f"MAE: ${mae:,.0f}")
print(f"R²: {r2:.4f}")

# 11. Save model
trainer.save_model("ft_transformer_model.pt")
```

## 📊 Expected Output

After training, you should see:
- Training progress with early stopping
- Validation metrics (RMSE, MAE, R²)
- Comparison with other models
- Model logged to MLflow

## 🎯 Where to Add in Your Notebook

Add FT-Transformer implementation **after Section 4: Data Cleaning and Preprocessing** and **before Section 8: Model Training and Comparison**.

This allows you to:
1. Preprocess data specifically for FT-Transformer
2. Train the model
3. Include results in the model comparison table

## 🔍 Integration with Existing Code

To add FT-Transformer results to your existing comparison table, add this after training:

```python
# Add FT-Transformer to comparison
ft_results = {
    'Model': 'FT-Transformer',
    'Train_RMSE': train_rmse_ft,
    'Val_RMSE': val_rmse_ft,
    'Train_MAE': train_mae_ft,
    'Val_MAE': val_mae_ft,
    'Train_R2': train_r2_ft,
    'Val_R2': val_r2_ft,
}

# Append to existing results
all_results.append(ft_results)

# Re-create comparison table
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('Val_R2', ascending=False).reset_index(drop=True)
print(results_df)
```

## 🚀 Performance Tips

1. **Batch Size**: Start with 256, increase if you have enough memory
2. **Learning Rate**: 0.001 works well, can try 0.0001 for fine-tuning
3. **Early Stopping**: Patience of 10 prevents overfitting
4. **Token Dimension**: 192 is good default, can increase to 256 for complex data
5. **GPU**: Significantly faster (10-50x) if available

## 🐛 Troubleshooting

### Import Error: No module named 'torch'
```python
!pip install torch
```

### Import Error: No module named 'rtdl'
```python
!pip install rtdl
```

### CUDA Out of Memory
- Reduce batch_size to 128 or 64
- Reduce d_token to 128
- Use CPU instead: `device='cpu'`

### Poor Performance
- Increase n_epochs to 100
- Try different learning rates: [0.0001, 0.001, 0.01]
- Adjust model architecture (n_blocks, d_token)

## 📈 Hyperparameter Tuning

You can tune these parameters in the config:
```python
config = {
    'd_token': 192,           # Try: [128, 192, 256]
    'n_blocks': 3,            # Try: [2, 3, 4]
    'attention_n_heads': 8,   # Try: [4, 8, 16]
    'attention_dropout': 0.2, # Try: [0.1, 0.2, 0.3]
    'ffn_d_hidden': 256,      # Try: [128, 256, 512]
    'ffn_dropout': 0.1,       # Try: [0.0, 0.1, 0.2]
}
```

## 📚 References

- **Paper**: "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021)
- **RTDL GitHub**: https://github.com/yandex-research/rtdl
- **PyTorch**: https://pytorch.org/

## ✅ Checklist

- [ ] PyTorch installed
- [ ] RTDL installed
- [ ] Data preprocessed for FT-Transformer
- [ ] Model built and configured
- [ ] Model trained with early stopping
- [ ] Results evaluated and compared
- [ ] Model saved
- [ ] MLflow tracking enabled

## 💡 Key Advantages of FT-Transformer

1. **Handles Mixed Data Types**: Seamlessly processes both numerical and categorical features
2. **No Manual Feature Engineering**: Learns feature interactions automatically
3. **State-of-the-Art Performance**: Often outperforms traditional ML models on tabular data
4. **Interpretability**: Attention weights can show feature importance
5. **Scalability**: Efficient on large datasets with GPU acceleration

## 🎓 Next Steps

After implementing FT-Transformer:
1. Compare performance with XGBoost/LightGBM
2. Try hyperparameter tuning with different configurations
3. Analyze attention weights for interpretability
4. Use the best model for final predictions
5. Document findings in MLflow

---

Happy modeling! 🚀
