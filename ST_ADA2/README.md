# ST_ADA2

l_src: labeled source, l_trg: labeled target, unl_trg: unlabeled target


## ST_ADAの問題点
1. D_loss (Discriminator用のloss) の計算時に，1batchを占める割合が l_src:l_trg:unl_trg = 2:1:1 になっていた．(l_trg:unl_trgが1:1だと，偏りのあるサンプリングとなる)

2. D_lossの計算において，l_src, l_trgではクラスバランスされているのに対し，unl_trgではバランスされていない．

3. D_lossにとっては，batchサイズが小さいことは問題点


## ST_ADAからの改善点
1. D_lossの計算には，unl_trgを使用しない．l_src:unl_trgを1:1の割合で使用．

2. D_lossの計算には，l_src, unl_trgを"クラスターバランシング"　→ D_loss用に別途Dataloaderを用意

3. batchサイズを大きくするために，RTX3090上で学習

