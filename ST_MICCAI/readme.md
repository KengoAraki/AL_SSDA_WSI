## Domain Adaptation with Active Learning
### MICCAI用の実験
### Source + single Target

クラスターのentropyを基準にターゲットのWSIを１枚選択し、訓練データに加えて学習する

- cluster entropyがMax, Med, Minの周辺5枚をそれぞれlabeled targetとする
- 残りのWSIからvalidationデータにはランダムに20枚割当
- validation用に割当後の残りがtest用 (unlabeled target)