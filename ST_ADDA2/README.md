# ST_ADDA2
l_src: labeled source, l_trg: labeled target, unl_trg: unlabeled target

## ADDA: Adversarial Discriminatrive Domain Adaptation
- sourceの特徴分布と識別境界は固定し，targetの特徴分布のみを遷移
- source用CNNとClassifierは固定．target用のCNNのみ更新
- source用CNNから抽出した特徴量 (l_srcの特徴量) とtarget用CNNから抽出した特徴量 (unl_trgの特徴量) をDiscriminatorに入力

### 損失関数:
- Discriminator Loss: l_srcとunl_trgで
- Adversarial Loss: unl_trgで
- Classification Loss?: l_trgでtarget用CNNを更新．(ただし，Classifierは固定)


1. D_lossの計算には，unl_trgを使用しない．l_src:unl_trgを1:1の割合で使用．

2. D_lossの計算には，l_src, unl_trgを"クラスターバランシング"　→ D_loss用に別途Dataloaderを用意

3. batchサイズを大きくするために，RTX3090上で学習

