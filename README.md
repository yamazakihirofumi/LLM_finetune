### This repo is my experience about fine tune an LLM to make it get better performance for specific task

Specifically, my friend in sociology department need some help on his project. he want to experience LLM and wish there's a LLM can run locally which help him distinguish emotional status in the online comment and that's why I start these experience
I'm writting this done as record and in case if someone wanna try replicate these experience by themself. 


#### The work flow: 
The raw data I have are .xlsx file include online comments, first after get the data I will do these steps

##### Data pre process:
0: environment set up:
install_env.sh : Run this to set up a local environment on your computer make sure you have anaconda installed 

1. excel_2_csv.py: Convert excel file to the csv file for easier process  

2.(Optional) 
batch_infer_gen.py: Since og datas are unlabeled, I choose to use bigger size model to help label these datas, specifically I choose to use DeepSeek-V3 651B help me lable the data, It work as teacher model here. 
Silicon flow provide batch process for cheaper price, run batch_infer_gen.py to help generate .jsonl file to send to silicon flow and get batched result. 
3.(Optional)
jsonl_to_csv.py:  Since the returned result only contain user_id so another script to merge two file to one csv file. it will generate the final labeled data for me, it will generat three file 
unlabeled_v2.csv, labeled_v2.csv and purged_labeled_v2. They all in these field "custom_id","message","class"
unlabeled_v2.csv: Unlabeled, only have custom_id and message, class inited as -1 
labeled_v2.csv : Merged the infrance result from two file inf_batch_v2.jsonl and silver_label_v2.jsonl , might contains some class = -1 since there's chance infrance goes wrong
purged_labeled_v2.csv : Based on labeled_v2.csv remove data where class = -1

4. Get the model:
For this experience, I tried to use Qwen developed by Alibaba, available at 
git clone https://huggingface.co/Qwen/Qwen2.5-3B
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B


5. local_inference.py: Before start fine tune, run this to test the base model 

6. finetune.py : Run this file to fine tune the model (might take long time)

7. local_inference.py: Run the inference again to check the performance diff compare to last run


    
    
    