1. 运行openie.py生成包含实体的文件：train_new.txt, text_new.txt, dev_new.txt (已在conceptnet_zc文件夹中)。
2. 在conceptnet_zc文件夹中，读入：train.tsv, test.tsv, dev.tsv，knowledge graph,  运行 procession.py， 会生成一个dataset文件夹，里面包含 train, test, dev 的邻接矩阵和矩阵所对应的词语index。
3. 在my_codes 文件夹中运行run_task1.py。