# transformer_attention
これを実際に作って理解する  
[作って理解する Transformer / Attention](https://qiita.com/halhorn/items/c91497522be27bde17ce#%E5%9F%BA%E6%9C%AC%E7%9A%84%E3%81%AA-attention)

試しにsphinxで[ドキュメント](./docs/_build/index.html)


# attention.py
基本的なatttention機構とそれをパラレルに並べたものを実装

# common_layer.py
- Position-wise Feedforward Network
- ResidualNormalizationWrapper
- LayerNormalization  
を実装

# embedding.py
- TokenEmbedding
- AddPositionalEncoding  
を実装