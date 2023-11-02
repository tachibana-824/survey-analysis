import MeCab
import pandas as pd
from gensim.models import Word2Vec

# CSVファイルからデータを読み込む関数
def load_csv(file_path, column_name):
    df = pd.read_csv(file_path)
    df_filtered_column = df[column_name].dropna()
    return df_filtered_column.tolist()

# テキストデータを形態素解析し、単語リストを返す関数
def tokenize_text(text_data_series):
    tagger = MeCab.Tagger('-d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd -r /opt/homebrew/etc/mecabrc')
    tagger.parse('')  
    tokenized_texts = []
    
    for text in text_data_series:
        node = tagger.parseToNode(str(text))  # 文字列として渡す
        words = []
        while node:
            features = node.feature.split(",")
            if features[0] not in ["BOS/EOS", "助詞", "助動詞", "記号"]:
                words.append(node.surface)
            node = node.next
        tokenized_texts.append(words)  # 単語のリストを返す
    
    return tokenized_texts


# Word2Vecモデルのトレーニングと類似単語の検索
def train_word2vec(sentences):
    # Word2Vecモデルのトレーニング
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    
    # トレーニング済みモデルの保存
    model.save("word2vec.model")
    
    # 類似単語の検索例
    target_word = 'プログラミング'
    if target_word in model.wv.key_to_index:
        similar_words = model.wv.most_similar(target_word, topn=10)
        print(f"{target_word}に類似した単語:")
        for word, score in similar_words:
            print(f"{word}: {score}")
    else:
        print(f"{target_word}は語彙に存在しません。")

if __name__ == "__main__":
    # ファイルとカラムの設定
    file_name = 'all-after'
    file_path = f'data/{file_name}.csv'
    column_name = 'Q8. この講義に対するコメントを自由に記載してください(自由記述)'
    
    # データ読み込み
    text_data = load_csv(file_path, column_name)
    
    # テキストデータの形態素解析
    sentences = tokenize_text(text_data)

    # Word2Vecで単語の埋め込みを学習し、類似単語を検索
    train_word2vec(sentences)

