import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

# データの読み込みと前処理
# データの読み込み
file_path = "izumi-before.csv"  # ファイルパスを適宜変更してください
data = pd.read_csv(file_path, encoding='utf-8')

# Pythonの理解度を数値に変換
understanding_mapping = {
    'わからない': 1,
    'わかる・低下': 2,
    'わかる・不変': 3,
    'わかる・向上': 4,
    'わかる・新規': 5
}

data['Python理解度_数値'] = data['Python理解度チェック：各項目を理解できるか教えて下さい。なお、資料の説明内容を全て理解していることは問うておりませんので、各項目の言わんとすることを理解できるか自己評価して下さい。\n\nわからない：学習前後を通じて理解できていない\nわかる・低下：学習前から理解しているが，理解度が低下した\nわかる・不変：学習前から理解しており，理解度は特に変わらなかった\nわかる・向上：学習前から理解しており，理解度が向上した\nわかる・新規：学習前は理解していなかったが，新たに理解できるようになった [モジュール]'].map(understanding_mapping)

# 学習方法を表す4つの特徴量を作成
learning_methods = ['プログラム実行', '動画視聴', '学習しなかった', '資料参照']

# 各学習方法に対して、バイナリ特徴量を作成
for method in learning_methods:
    data['学習方法: ' + method] = data['学習項目：教材の各項目をどのように学習したか教えて下さい（複数回答可） [モジュール]'].apply(lambda x: 1 if method in x else 0)

# 特徴量とターゲットを定義
X = data[['学習方法: ' + method for method in learning_methods]]  # 特徴量
y = data['Python理解度_数値']  # ターゲット



# データの準備
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)




#線形回帰
# モデルの定義
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# モデル、損失関数、オプティマイザの設定
model = LinearRegressionModel(X_tensor.shape[1])
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# モデルのトレーニング
num_epochs = 1000
for epoch in range(num_epochs):
    # フォワードパス
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # バックワードパスとオプティマイザのステップ
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 進捗の表示
    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# トレーニングされたモデルのパラメータ
params = list(model.parameters())
weights = params[0].data.numpy().flatten()
bias = params[1].data.numpy().flatten()

print(weights, bias)



# # 学習方法のカラムをワンホットエンコーディング
# learning_method_encoded = data['学習項目：教材の各項目をどのように学習したか教えて下さい（複数回答可） [モジュール]'].str.get_dummies(sep='、')
# print(learning_method_encoded)
# learning_method_encoded.columns = ['学習項目：教材の各項目をどのように学習したか教えて下さい（複数回答可） [モジュール]: ' + col for col in learning_method_encoded.columns]
# print(learning_method_encoded.columns)

# # データフレームにエンコードされたカラムを結合
# data_encoded = pd.concat([data, learning_method_encoded], axis=1)
# print(data_encoded)

# X = data_encoded[learning_method_encoded.columns]  # 特徴量: 学習方法のダミー変数
# y = data_encoded['Python理解度_数値']  # ターゲット: Pythonの理解度スコア

# 非線形回帰
# # モデルの定義
# class NonlinearRegressionModel(nn.Module):
#     def __init__(self, input_dim):
#         super(NonlinearRegressionModel, self).__init__()
#         self.linear1 = nn.Linear(input_dim, 5)  # 中間層のニューロン数を5とする
#         self.linear2 = nn.Linear(5, 1)  # 出力層のニューロン数を1とする

#     def forward(self, x):
#         x = self.linear1(x)  # 最初の線形層
#         x = F.relu(x)  # ReLU活性化関数
#         x = self.linear2(x)  # 2つ目の線形層
#         return x
    

# # モデル、損失関数、オプティマイザの設定
# model = NonlinearRegressionModel(X_tensor.shape[1])  # ここを変更
# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001)

# # モデルのトレーニング
# num_epochs = 1000
# for epoch in range(num_epochs):
#     # フォワードパス
#     outputs = model(X_tensor)
#     loss = criterion(outputs, y_tensor)
    
#     # バックワードパスとオプティマイザのステップ
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # 進捗の表示
#     if (epoch+1) % 1000 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# # トレーニングされたモデルのパラメータ
# params = list(model.parameters())
# # 最初の線形層の重みとバイアス
# weights1 = params[0].data.numpy()
# bias1 = params[1].data.numpy()
# # 2つ目の線形層の重みとバイアス
# weights2 = params[2].data.numpy()
# bias2 = params[3].data.numpy()

# print("First linear layer weights and bias:", weights1, bias1)
# print("Second linear layer weights and bias:", weights2, bias2)

