
## Research question
Social equility discussion is a important topic for sociology, researchers often search internet for collecting data, but most of time the data from internet are un-censored and un-labeled. Which creat barrier for researchers to analyze and organize datas. In this work I collaborate with student in sociology department and trained a LLM based on the thing. This work is focus on technical part, include how to how to cleaning data, how to use data we already have to fine-tune the fundation model we have, how to make fine tune process more efficient with hardware - awareness. What we not do include: Prompt engineering to let model generate output, data analysis include justify if every data we used for training/testing are accurate. Since they require domain specific work. Also as requested by data provider,we are not including original data or internediate data since they are not public data. But if you want to replicate the work or collaborate on that. Please contact me or request by github issue. 



## Project work flow: 
On the big picture this project work flow is first the data will be transformed from .xsls to .csv for easier process in the future.
Then the data will be delivered to large size LLM which have better performance to be labled(In this project I choose DeepSeek V3 651B). After the data been labled I use that to train our small model and after the training is done, test it's performance. 

##### Program introduction:
0: environment set up:
install_env.sh : Run this to set up a local environment on your computer make sure you have anaconda installed 

1. excel_2_csv.py: Convert excel file to the csv file for future process  

2.(Optional) 
batch_infer_gen.py: Since many online platform support batch inference. To make data label more automatic and done in batches. Here this program basically concatinate prompt.txt with every line of data entry in csv file to generate one single inference job. Since there are around 70k lines of data in original file, there will be around 70K inference request contained in the single file. 
Batch inference service provider here I choose silicon flow but it should work for any other service provider.  
In summary it's 
./pre_process/datas/inf_batch_v2.jsonl --[Online service provider] --> ./pre_process/datas/silver_label_v2.jsonl

3.(Optional)
jsonl_to_csv.py:  Since the returned result only contain user_id so another script to merge two file to one csv file. it will generate the final labeled data for me, it will generat three file 
unlabeled_v2.csv, labeled_v2.csv and purged_labeled_v2. They all in these field "custom_id","message","class"
unlabeled_v2.csv: Unlabeled, only have custom_id and message, class inited as -1 
labeled_v2.csv : Merged the infrance result from two file inf_batch_v2.jsonl and silver_label_v2.jsonl , might contains some class = -1 since there's chance infrance goes wrong
purged_labeled_v2.csv : Based on labeled_v2.csv remove data where class = -1

4. Get the model:
For to make this github repo clean, I didn't upload the model, they are available at hugging face. You can get them by run git clone command in this directory

git clone https://huggingface.co/Qwen/Qwen2.5-3B
#git clone https://huggingface.co/Qwen/Qwen2.5-0.5B


5. (Optional)local_inference.py:
This file is used to run infrance jobs locally. it take prompt, model and data as test.csv as input. It's used to test a single model, change the local_model_path in this file to change model you want to use 

6. finetune.py
This is the python file I used to fine tune the model. It will use qlora to fine tune the model, and save the checkpoint to the path. 


7. posttune_eval.py
This file will take both base model and LoRA adapter and run the infrance based on that. 




#### Bert 
Since this specific task is classification task, so I also try the whole work flow based on Bert, specifically bert-base-chinese model. It can be download from 
git clone https://huggingface.co/google-bert/bert-base-chinese
Since Bert have different model family than Qwen2.5, we choose use similar but not exact same code to infrance and evaluate, but the workflow are similar, see these following files
{bert_eval.py, bert_tune.py, bert_tune_eval.py,bert_inference_compare.py}. We also include the evaluation.


## Observation and conclusions :
For this project I tired differnet model combination include bert-base-chinese(110M), Qwen2.5-3B, and Qwen2.5-0.5B. I started my approach from Qwen2.5 models since it's popular recently and more flexible for potential future job(Like comment summarization and emotional analysis). But since the task right now it just classification, I also run experiments on bert. 
Surpringly, even bert have relative way much smaller parameter size. It's outperform Qwen2.5-0.5B and even slitely better than Qwen2.5-3B. 
Also, another problem of decoder only architecture is they have randonness when generating outputs. for example even I ask it give me result like "output result as '[1]' " It might still generate like '(1)' thought it have a correct preduction. And that brings in another problem that the result might meaningful but not useful. it might generate result like 'It seems like this comment do not have gender bias' but we are expecting it return result as one single number from 0-3 reflecting it's class. As we can see from Qwen2.5-3B base model that's a huge problem since valid predictions are only 42.33% which makes the accuracy only around 16% (0.3753 * 42.33%) with that number is even worse than random-guessing. Though the validation become near 100% after fine tune, but accuract is still not better than bert. 






## Evaluations

#### result for Qwen2.5-3B
--------------------------------------------------------------------------------
Model: ./Qwen2.5-3B
Test set: ./pre_process/evaluation/test.csv
Total samples: 14849
Valid predictions: 6285 (42.33%)
Accuracy: 0.3753

Classification Report:
              precision    recall  f1-score   support

    Positive       0.10      0.12      0.11       888
        Mild       0.55      0.63      0.59      2881
    Negative       0.18      0.34      0.23       526
  Irrelevant       0.29      0.14      0.19      1990

    accuracy                           0.38      6285
   macro avg       0.28      0.31      0.28      6285
weighted avg       0.37      0.38      0.36      6285

Base Model: ./Qwen2.5-3B
LoRA Adapter: ./qlora_checkpoints/final_model
Test set: ./pre_process/evaluation/test.csv
Total samples: 14849
Valid predictions: 14849 (100.00%)
Accuracy: 0.7210

Classification Report:
              precision    recall  f1-score   support

    Positive       0.44      0.82      0.58      1944
        Mild       0.84      0.76      0.80      6377
    Negative       0.53      0.47      0.50      1313
  Irrelevant       0.85      0.70      0.77      5215

    accuracy                           0.72     14849
   macro avg       0.67      0.69      0.66     14849
weighted avg       0.76      0.72      0.73     14849


#### Evaluation result for Qwen2.5-0.5B
--------------------------------------------------------------------------------

Model: ./Qwen2.5-0.5B
Test set: ./pre_process/evaluation/test.csv
Total samples: 14849
Valid predictions: 11826 (79.64%)
Accuracy: 0.1490

Classification Report:
              precision    recall  f1-score   support

    Positive       0.17      0.72      0.28      1581
        Mild       0.44      0.02      0.03      5064
    Negative       0.08      0.37      0.13      1019
  Irrelevant       0.36      0.04      0.07      4162

    accuracy                           0.15     11826
   macro avg       0.26      0.29      0.13     11826
weighted avg       0.35      0.15      0.09     11826

Base Model: ./Qwen2.5-0.5B
LoRA Adapter: ./qlora_checkpoints_0.5B/final_model
Test set: ./pre_process/evaluation/test.csv
Total samples: 14849
Valid predictions: 14849 (100.00%)
Accuracy: 0.6532

Classification Report:
              precision    recall  f1-score   support

    Positive       0.42      0.76      0.54      1944
        Mild       0.73      0.75      0.74      6377
    Negative       0.41      0.37      0.39      1313
  Irrelevant       0.81      0.56      0.67      5215

    accuracy                           0.65     14849
   macro avg       0.60      0.61      0.59     14849
weighted avg       0.69      0.65      0.66     14849




#### Evaluation on Bert
--------------------------------------------------------------------------------
Before fine tune
Test accuracy: 0.7304

Classification report:
              precision    recall  f1-score   support

           0       0.85      0.72      0.78      1944
           1       0.82      0.76      0.79      6377
           2       0.56      0.18      0.27      1313
           3       0.64      0.83      0.72      5215

    accuracy                           0.73     14849
   macro avg       0.72      0.62      0.64     14849
weighted avg       0.74      0.73      0.72     14849


After fine tune
Test accuracy: 0.8232

Classification report:
              precision    recall  f1-score   support

           0       0.85      0.84      0.85      1944
           1       0.89      0.86      0.87      6377
           2       0.58      0.50      0.54      1313
           3       0.79      0.86      0.82      5215

    accuracy                           0.82     14849
   macro avg       0.78      0.76      0.77     14849
weighted avg       0.82      0.82      0.82     14849

    
    
    