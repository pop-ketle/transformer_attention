# transformer_attention
これを実際に作って理解する  
[作って理解する Transformer / Attention](https://qiita.com/halhorn/items/c91497522be27bde17ce#%E5%9F%BA%E6%9C%AC%E7%9A%84%E3%81%AA-attention)

試しにsphinxで[ドキュメント](https://pop-ketle.github.io/transformer_attention/)

# 感想
写経してみたけど正直あまり理解度は深まらなかった。多分Qiitaの記事をみて理解してくれってことなんだろうけどし正直もっとコード中にコメントで解説を書いてほしかった。あんまりコードを写経する意味はなく、他の解説とかも含めて解説記事読んでる方が理解が深まり、断然マシだった。  
ついでに自身のTensorFlowについての理解がかなり浅いことも分かった。Keras触ってるからなんとなくわかるだろうと思っていたけど。

Sphinxもかなりわかりにくいところが多く、下記のサイトを見つけるまでなかなかドキュメント製作すらまともにできなかった。(ついでにGithub Pagesの更新が遅くてなかなか適応されず、自分がおかしいのかと何度かやり直した。)
忘れないように参考までにメモ
https://qiita.com/futakuchi0117/items/4d3997c1ca1323259844

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

# transformer.py
- Transformer model
- Encoder model
- Decoder model  
を実装