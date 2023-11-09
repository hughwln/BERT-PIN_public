# BERT-PIN_public

This is the source code of BERT-PIN, a method to restore missing data segments in load profiles.The model uses BERT-based model to learn the load pattern and achive a high restoration accuracy. The model is desined as follow:

![image](https://github.com/hughwln/BERT-PIN_public/assets/20769362/0aa460c0-d181-4a49-9529-d080272be492)

Examples of load profile inpainting:

![image](https://github.com/hughwln/BERT-PIN_public/assets/20769362/98eddf47-81f4-4e86-8a0b-96f87848d1a3)

For more details, please refer to our paper.

### Citation
If you use the code, Please cite this paper.

Yi Hu, Kai Ye, Hyeonjin Kim, and Ning Lu, “BERT-PIN: A BERT-based Framework for Recovering Missing Data Segments in Time-series Load Profiles”, available at http://arxiv.org/abs/2310.17742


### Note: 
The training data is not included in this repository, due to privacy issue. You can try to run the code with your own data. 

### Contact:
Please send questions or comments to Yi Hu at hugh19flyer@gmail.com

### Directory:
The single patch daily version of BERT-PIN model is built in src_bert/model.py and trained in src_bert/train.py

The multiple patch weekly version of BERT-PIN model is built in src_multipatch/model.py and trained in src_multipatch/train.py
