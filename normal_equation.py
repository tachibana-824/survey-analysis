import pandas as pd
import numpy as np
import numpy as np

# データの読み込みと前処理
file_name = "nisui-before.csv"
file_path = f"data/{file_name}"
data = pd.read_csv(file_path, encoding='utf-8')

# Pythonの理解度を数値に変換
understanding_mapping = {
    'わからない': 1,
    'わかる・低下': 2,
    'わかる・不変': 3,
    'わかる・向上': 4,
    'わかる・新規': 5
}

data_understandings = ['Python理解度チェック：各項目を理解できるか教えて下さい。なお、資料の説明内容を全て理解していることは問うておりませんので、各項目の言わんとすることを理解できるか自己評価して下さい。\n\nわからない：学習前後を通じて理解できていない\nわかる・低下：学習前から理解しているが，理解度が低下した\nわかる・不変：学習前から理解しており，理解度は特に変わらなかった\nわかる・向上：学習前から理解しており，理解度が向上した\nわかる・新規：学習前は理解していなかったが，新たに理解できるようになった [クラス]',
                      'Python理解度チェック：各項目を理解できるか教えて下さい。なお、資料の説明内容を全て理解していることは問うておりませんので、各項目の言わんとすることを理解できるか自己評価して下さい。\n\nわからない：学習前後を通じて理解できていない\nわかる・低下：学習前から理解しているが，理解度が低下した\nわかる・不変：学習前から理解しており，理解度は特に変わらなかった\nわかる・向上：学習前から理解しており，理解度が向上した\nわかる・新規：学習前は理解していなかったが，新たに理解できるようになった [モジュール]',
                      'Python理解度チェック：各項目を理解できるか教えて下さい。なお、資料の説明内容を全て理解していることは問うておりませんので、各項目の言わんとすることを理解できるか自己評価して下さい。\n\nわからない：学習前後を通じて理解できていない\nわかる・低下：学習前から理解しているが，理解度が低下した\nわかる・不変：学習前から理解しており，理解度は特に変わらなかった\nわかる・向上：学習前から理解しており，理解度が向上した\nわかる・新規：学習前は理解していなかったが，新たに理解できるようになった [パッケージ]'
                      ]

sh_data_understandings = ['クラス', 'モジュール', 'パッケージ']

data_methods = ['学習項目：教材の各項目をどのように学習したか教えて下さい（複数回答可） [クラスの一般的表記]',
                '学習項目：教材の各項目をどのように学習したか教えて下さい（複数回答可） [モジュール]',
                '学習項目：教材の各項目をどのように学習したか教えて下さい（複数回答可） [パッケージ]'
                ]

sh_data_methods = ['クラスの一般的表記', 'モジュール', 'パッケージ']

learning_methods = ['プログラム実行', '動画視聴', '学習しなかった', '資料参照']

# 結果を保存するための空のDataFrameを作成
results_df = pd.DataFrame()

for i in range(len(data_understandings)):
    understanding_col = data_understandings[i]
    method_col = data_methods[i]
    sh_data_understanding = sh_data_understandings[i]
    sh_data_method = sh_data_methods[i]
    
    # Python理解度を数値に変換
    data[understanding_col + '_数値'] = data[understanding_col].map(understanding_mapping)
    
    # 学習方法の特徴量作成
    for method in learning_methods:
        data[method_col + ': ' + method] = data[method_col].apply(lambda x: 1 if method in x else 0)
        
    # 特徴量とターゲットの定義
    X = data[[method_col + ': ' + method for method in learning_methods]]
    y = data[understanding_col + '_数値']
    
    # 正規方程式を使用して重みを計算
    X_matrix = np.hstack([np.ones((X.shape[0], 1)), X.values])
    theta_normal_equation = np.linalg.pinv(X_matrix.T @ X_matrix) @ X_matrix.T @ y.values
    eigenvalues, eigenvectors = np.linalg.eig(X_matrix.T @ X_matrix)
    print(f"固有値： {eigenvalues}")
    bias_normal_equation = theta_normal_equation[0]
    weights_normal_equation = theta_normal_equation[1:]
    
    print(f"Correlation between {sh_data_understanding} and {sh_data_method}:")
    print("Weights:", weights_normal_equation)
    print("Bias:", bias_normal_equation)
    print("---------------")

    # 重みとバイアスをpandasのDataFrameに変換
    temp_df = pd.DataFrame({
        'Understanding': sh_data_understanding,
        'Method': sh_data_method,
        'Feature': ['Bias'] + [method for method in learning_methods],
        'Weight': np.concatenate(([bias_normal_equation], weights_normal_equation))
    })
    
    # 結果をまとめるDataFrameに追加
    results_df = pd.concat([results_df, temp_df], ignore_index=True)

    # 空行を追加
    empty_row = pd.DataFrame({column: [np.nan] for column in results_df.columns})
    results_df = pd.concat([results_df, empty_row], ignore_index=True)

# DataFrameをCSVファイルに保存
output_path = f"result/{file_name.replace('.csv', '_results.csv')}"

results_df.to_csv(output_path, index=False, na_rep='')
print("Saved all the weights and biases to", output_path)

