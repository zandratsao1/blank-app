import joblib
# 加载你现在本地那个大的模型文件
model = joblib.load('rf_model.pkl')
# 重新保存，compress=3 是一个很好的平衡点（1-9可选）
joblib.dump(model, 'rf_model.pkl', compress=3)