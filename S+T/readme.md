## Domain Adaptation with Active Learning
### Source + single Target

クラスターのentropyを基準にターゲットのWSIを１枚選択し、訓練データに加えて学習する

1. ターゲットデータのみを用いてクラスタリングを行う
2. entropyの高いWSIを１枚選択し、アノテーションする
3. 2のWSIをパッチに切り取り、訓練データに加えて再学習する

- クラスタリング手法: k-means++ \
- 特徴抽出手法: ソースデータで学習したResnet-50から抽出した特徴量を、PCAで30次元に削減 \
- entropy: WSIのパッチがどれだけ多くのクラスターに所属しているかを表す \

```
MF0013 wsi_num: 108
== min ==
0.5047_03_G109-1

== first quartile ==
1.8215_03_G45

== med ==
2.0860_03_G293

== third quartile ==
2.2942_03_G18

== max ==
3.0519_03_G144
```
