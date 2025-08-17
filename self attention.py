import numpy as np

def softmax(x, axis):
    """Compute softmax along specified axis with numerical stability."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)  # 防止指数溢出
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Q, K, V matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    """Compute scaled dot-product self-attention."""
    d_k = Q.shape[-1]  # 键向量维度
    # 修正：使用批量矩阵乘法和缩放因子
    attention_scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    # 修正：在最后一维上应用softmax
    attention_weights = softmax(attention_scores, axis=-1)
    attention_output = np.matmul(attention_weights, V)
    #print("注意力权重和:", np.sum(attention_weights[0, 0]))  # 应接近1.0
    return attention_output

# 示例验证
batch_size = 2
seq_len = 3
hidden_dim = 4

X = np.random.randn(batch_size, seq_len, hidden_dim)
W_q = np.random.randn(hidden_dim, hidden_dim)
W_k = np.random.randn(hidden_dim, hidden_dim)
W_v = np.random.randn(hidden_dim, hidden_dim)

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output = self_attention(Q, K, V)

