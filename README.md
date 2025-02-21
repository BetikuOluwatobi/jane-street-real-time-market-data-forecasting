---

# Jane Street Kaggle Competition Solution

This repository contains my solution to the **Jane Street Kaggle Competition**, where the goal was to build machine learning models using real-world financial data derived from production systems. The challenge focused on modeling complex financial markets, highlighting difficulties such as fat-tailed distributions, non-stationary time series, and sudden market behavior shifts.

---

## 📚 Problem Description

The competition required participants to model financial market data and predict responder values, tackling challenges like:

- **Non-stationary time series**  
- **Fat-tailed distributions**  
- **Dynamic market behaviors**  

The anonymized and lightly obfuscated dataset offered insights into trading strategies and modern financial challenges.

---

## 🛠️ Solution Overview

This project applied **transformer architectures** to model financial time-series data effectively. While initially exploring **LightGBM**, deep learning architectures significantly outperformed traditional boosting methods. 

### Key Highlights:
1. **Large Dataset Handling**: Leveraged PyTorch's lazy loading to manage datasets that couldn't fit into memory.  
2. **Transformer Architectures**:  
   - **Multi-Head Attention with Linear Biases**  
   - **Multi-Head Attention without Linear Biases**  
3. **Minimal Feature Engineering**: Focused on model performance rather than feature engineering.  
4. **Model Comparison**: Achieved 4x performance improvement over **LightGBM**.

---

## 🚀 Model Architectures

### Multi-Head Attention with Linear Biases
Features an **attention mechanism** with relative position encoding for improved time-series modeling. Includes custom **LayerNorm** and feedforward layers.

### Multi-Head Attention without Linear Biases
A simplified version without linear biases, designed to compare the impact of position-based attention mechanisms.

### Key Code Snippets:
```python
# Sample attention mechanism with linear biases
attn_scores = (queries @ keys.mT) / self.attn_scale + self.bias
attn_scores = attn_scores.masked_fill(self.mask, -torch.inf)
attn_weights = self.softmax(attn_scores)
```

---

## 🧪 Results

- **Cross-Validation (CV)**: Transformer architectures consistently outperformed boosting algorithms.  
- **Leaderboard Performance**: Both architectures showed improved scores on the public leaderboard, validating the effectiveness of deep learning for financial modeling.  

---

## 📈 Competition Insights

This competition was an excellent opportunity to:
- Apply concepts from **Sebastian Raschka's book** on building LLMs from scratch.
- Explore deep learning's application to **time-series financial data**.
- Understand the practical challenges of **large-scale dataset handling**.

---

## 📋 Key Takeaways

1. **Deep Learning for Financial Data**: Demonstrated significant performance improvements over traditional models.  
2. **Minimal Feature Engineering**: Focused solely on model performance using selected features from **LightGBM**.  
3. **Practical Challenges**: Managed memory-intensive datasets through efficient data loading strategies.

---

## 🔗 Resources

- **Competition Details**: [Jane Street Kaggle Competition](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/)  
- **Sebastian Raschka's Book**: [Link](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167)  

---

## 💡 Future Work

- Experiment with hybrid architectures combining boosting algorithms and transformers.  
- Optimize transformer models for inference speed in real-world trading systems.

---

Feel free to explore the code and reach out with any feedback or suggestions! 😊  

--- 

## 🖋️ Author

**Oluwatobi Betiku**  
- 📧 Email: [betikuoluwatobi7@gmail.com](mailto:betikuoluwatobi7@gmail.com)  
- 🌐 LinkedIn: [Oluwatobi Betiku](https://www.linkedin.com/in/oluwatobi-betiku-oluwatobi/)  
- 🐦 Kaggle: [Oluwatobi's Profile](https://www.kaggle.com/oluwatobibetiku)  

--- 
