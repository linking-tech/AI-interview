import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    位置编码模块，为输入序列添加位置信息
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数索引使用正弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数索引使用余弦函数
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # 不参与训练的参数
    
    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """
    多头注意力模块
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch_size, seq_len, d_model)
        batch_size = query.size(0)
        
        # 线性变换并分成多头
        # (batch_size, num_heads, seq_len, d_k)
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch_size, num_heads, seq_len, seq_len)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attn = self.dropout(attn)
        
        # 应用注意力到值
        output = torch.matmul(attn, v)  # (batch_size, num_heads, seq_len, d_k)
        
        # 拼接多头结果
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)
        
        # 最终线性变换
        output = self.w_o(output)
        
        return output, attn

class PositionWiseFeedForward(nn.Module):
    """
    位置-wise前馈网络
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.GELU()  # 使用GELU激活函数，原论文使用的是ReLU
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    """
    Transformer编码器层
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        # 自注意力子层
        attn_output, attn = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x, attn

class DecoderLayer(nn.Module):
    """
    Transformer解码器层
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)  # 掩码自注意力
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)  # 编码器-解码器注意力
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, self_mask, cross_mask):
        # 掩码自注意力子层
        attn_output, self_attn = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 编码器-解码器注意力子层
        attn_output, cross_attn = self.cross_attn(x, enc_output, enc_output, cross_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        
        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x, self_attn, cross_attn

class Encoder(nn.Module):
    """
    Transformer编码器
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # 编码器层堆叠
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        
        # 嵌入层 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)  # 缩放嵌入
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # 经过所有编码器层
        attn_list = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attn_list.append(attn)
        
        return x, attn_list

class Decoder(nn.Module):
    """
    Transformer解码器
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # 解码器层堆叠
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层，用于预测下一个token
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, enc_output, self_mask, cross_mask):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        
        # 嵌入层 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)  # 缩放嵌入
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # 经过所有解码器层
        self_attn_list = []
        cross_attn_list = []
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, enc_output, self_mask, cross_mask)
            self_attn_list.append(self_attn)
            cross_attn_list.append(cross_attn)
        
        # 输出层
        output = self.fc_out(x)  # (batch_size, seq_len, vocab_size)
        
        return output, self_attn_list, cross_attn_list

# 示例：创建Transformer编码器和解码器
if __name__ == "__main__":
    # 超参数
    vocab_size = 5000  # 词汇表大小
    d_model = 512      # 模型维度
    num_layers = 6     # 编码器和解码器层数
    num_heads = 8      # 注意力头数
    d_ff = 2048        # 前馈网络隐藏层维度
    max_len = 100      # 最大序列长度
    dropout = 0.1      # dropout概率
    
    # 创建编码器和解码器
    encoder = Encoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
    decoder = Decoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
    
    # 随机生成输入
    batch_size = 32
    src_seq_len = 20
    tgt_seq_len = 25
    
    src = torch.randint(0, vocab_size, (batch_size, src_seq_len))  # 源序列
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))  # 目标序列
    
    # 创建掩码（这里简化处理，实际应用中需要根据具体任务设计）
    src_mask = torch.ones(batch_size, 1, src_seq_len, src_seq_len)  # 源序列掩码
    tgt_self_mask = torch.ones(batch_size, 1, tgt_seq_len, tgt_seq_len)  # 目标序列自注意力掩码
    tgt_cross_mask = torch.ones(batch_size, 1, tgt_seq_len, src_seq_len)  # 编码器-解码器注意力掩码
    
    # 前向传播
    enc_output, enc_attn = encoder(src, src_mask)
    dec_output, dec_self_attn, dec_cross_attn = decoder(tgt, enc_output, tgt_self_mask, tgt_cross_mask)
    
    print(f"编码器输出形状: {enc_output.shape}")  # (batch_size, src_seq_len, d_model)
    print(f"解码器输出形状: {dec_output.shape}")  # (batch_size, tgt_seq_len, vocab_size)
