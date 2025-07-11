## Notes 01.04.2025
* I tried using XLM-RoBERTa model for checking sentence similarities. My final assumption is that _XLM-RoBERTa_ model is not designed for checking the sentence similarities. Therefore, it gave the poor results for **both** Azerbaijani and English languages.
* **sentence_transformers** library is performing better when comparing the sentence similarities rather that XLM-RoBERTa model. 

_I understood that for checking contextual embeddings of different words, I shouldn't check the sentence similarities._

* I decided to visualize and see the embeddings of polysemious words across the different sentences. For example, the word "bal" has 2 meanings: honey and score. There are 4 sentences, 2 of them contain "bal" word with the meaning of honey, the other 2 contain "bal" word with the meaning of score. 
* `comparison.py` script compares the embedding of each "bal" word in each sentence from the **1st**, **4th**, **8th** and **12th** layer of the transformer of XLM-RoBERTa model and shows similarity matrix heatmap. It also gives the PCA representation of each "bal" word in each sentence. Moreover, it shows the overall similarities (the similarities between embeddings derived from the **last**, **12th** layer of the transformer layer) between different models: **XLM-RoBERTa** and **m-BERT**. 
* This code is derived from the mainly Claude. I need to check this code more carefully for understanding it thouroughly. Then, I can improve it. 

**THE IMPORTANT NOTE**
* As initial visualization of results of `comparison.py` didn't show meaningful results. It had to show high similarities between the similar sentences and low similarity between distinct sentences. However, it almost showed high similarities between all sentences, accross different layers of the tansformer of the XLM-RoBERTa model. So, either I was doing something wrong or embeddings created by XLM-RoBERTa model perform poor. 
* For checking whether I am doing something wrong or contextual embeddings created by XLM-RoBERTa models are not proper, I created 4 sentences containing the polysemious word: **bank**. 2 of the sentences use "bank" word with the meaning of bank, the other 2 use "bank" wrod with the meaning of shore, coast. 
* `comparison.py` showed that, the contextual embeddings show high similarities in the similar sentences and less similarities in the different sentences which is expected output. 

From that point we can conclude:
**XLM-RoBERTa model performs poor in Azerbaijani language.**

* P.S. These are my initial assumptions, I need to clarify my assumptions. 

## Notes 08.04.2025
* I created a PDF document and created chunks of it. 
* Our main embedding model is **BGE-M3** model. I derived vector embeddings of chunked documents. Then, stored them inside the vector database. 
* Then, I wrote a query and derived embedding of this query, too.
* *cosine similarity* is used to calculate similarity score between query embedding and embeddings of chunked documents. 

**Result**
* BGE-M3 embedding model doesn't perform poor. Its results are reasonable. I only did initial observation and initial implementation. For getting accurate observation, I have to improve the script for processing documents and creating vector database, chunks, etc. 

## Notes 16.04.2025
* Created `processor.py` script which:
    * reads PDF document;
    * cleans and normalizes this PDF text;
    * splits it into chunks;
    * saves these chunks in the specified directory;
    * creates a ChromaDB vector database for storing vector embeddings 

* Created `smt.ipynb` => this notebook shows how to use `processor.py` script. Actually, there should be **main()** function for running *Preprocessor* class and its methods. This will be done

* Modified `preprocessor.ipynb` so that the irrelevant functions and code snippets have been removed. Although the main processor script is `processor.py`, this notebook will be used for trying new and additional staff in future works. 

### TODO:
* Create **main()** script for running all stuff: query is written by user and relevant documents are retrieved using processor script and embedding model. 
* Evaluate the performance of processor script for _different embedding models_ and _different PDFs_. If it works, this script may be used for all further processes. 

## Notes 17.04.2025
* I have tried **XLM-RoBERTa** model with `processor.py` script. It gave me an error when creating vector database. I should look at this error and fix it.

## Notes 18.04.2025
* I fixed an issue, when embeddings are torch tensors using **squeeze()** command. Embedding vectors are created as 2D torch tensors which was the reason why the processor script was giving an error when creating database. It is expecting 1D array (list), but we were passing 2D. **squeeze()** command makes this 2D tensor 1D.

* I have modified `smt.ipynb`so that it is more compact and structured way for trying different embedding models and different documents with different questions. 

* Created `sample_doc.pdf` document in Azerbaijani which is more structured and cleaner text document. We will use this document and relevant questions for evaluating the performance of different multi-lingual models.

## Notes 20.04.2025
* Modified *create_database()* method so that it saves the created databases in **permanent directories**. rather than temporary ones.

* Created 2 main notebooks:
    * `first_steps.ipynb` => Loading the given PDF document, reading, cleaning and normalizing the text inside the document, splitting it into chunks and saving these chunks, creating vector embeddings of these chunks and saving these vector embeddings inside the vector database.
    * `retrieval.ipynb` => Loading the vector database, embedding the query and retrieving the relevant document, and saving the result text in .txt file.

* Tried the 3rd embedding model: *LaBSE: Language Agnostic BERT Sentence Embeddings*. **This embedding model didn't perform well on Azerbaijani language.**

* Removed unnecessary notebooks from the project directory.

* **TODO**:
    * As because the necessary scripts have been written, I have to prepare final result from 3 or 5 embedding models with multiple queries. That is the main thing to do!

## Notes 23.04.2025
* Created `report.pdf` document that reports 3 outcomes of 3 different models with 3 different queries;

* `retrieval.ipynb` script was corrupted in the previous commit. I have re-written and re-commited it.

## Notes 01.06.2025
* Dataset files were in *.json* which was lists of dictionaries, but in documentation it has been written that training data should be just dictionaries, not list of them. I have fixed it, converted them to **.jsonl** files. 

* Some dataset files contained English prompts, I converted them to Azerbaijani propmpts

* Modified the overall structure of the project. Divided the overall project into 2 main folders: `data/` which contains dataset files, results and other documents, whereas `src/` folder contains source codes. 

### TODO:
* Should I create 2 datasets: validation data and training data for finetuning; or should I combine all dataset files into just 1 training data? *I should clarify this point.*

* Is it better to add `FlagEmbedding` directory to the main project directory? Should clarify this.

* Should make training process on 1, or maximum 2 days. Should create a training script. 

## Notes 02.06.2025
* Decided to use hard negatives for increasing the number of hard negatives for making model more robust and strong. However, miner script (`FlagEmbedding/scripts/hh_miner.py`) didn't work with my dataset, because format was not correct.
* The correct format of the dataset is so that each line consists of 1 full query: query, its positive and negative answers, scores, prompts and so on. All information about a query should be in 1 line not on separate lines. Therefore, I changed the structure of some dataset files. Most of them should be modified also. 

* Note that: **595iiq_dataset.jsonl** and **817_dataset.jsonl** files couldn't generate hard negatives. Should investigate it.

* The `_minedHN.jsonl` ending files are hard mined negatives.

## Notes 03.06.2025
* Modified all dataset files (*.jsonl* files) so that each line consists of 1 full query: query itself, positive and negative answers and prompt, **removed** _pos_scores and neg_scores_ , because distillation knowledge is not used.

* applied mining of hard negatives to all datasets and saved them in the directory of `data/mined_HN`.

* I tried to run training script of BGE-M3 model, but it was giving so much errors. I fixed all of them, but 1 of them remained problematic.

* **CUDA Out-Of-Memory** problem. It says that 5.5 GiB of GPU storage is used and 2.12 more GoB is needed for storing optimizer states, however, there is no such an available memory on GPU. I have to analyze this script from the very beginning and understand what is the problem. It looks like that tis is a significant problem and I need to address ASAP. 

## Notes 23.06.2025
* BGE-M3 Model couldn't be trained on my local machine (work computer) because of memory constraints. The model is too large to be trained on GPU with VRAM of 8 GB. 

* I had to train BGE-M3 model on AWS Server. There were some problems that made training procedure to lag, but it was trained finally. 

* The training has been done for **10** epochs with batch_size **2** per device. Maximum sequence length for query is **256**, whereas maximum sequence length for a passage was also **256**. Train group size was **4**. 

### TODO:
* Some training parameters for example (*train_group_size*) remained unclear to me. I have to look through and understand what each parameter does in the training. 

* Trained model is on the server. I copied it using `scp` command to my local folder (but in home computer). I uploaded this to Google drive for accessing on my work computer (I cannot connect to AWS Server at work due to restrictions). However, when I downloaded the files from Google Drive, the downloaded file's structure is different from the original. **I have to observe carefully what there is inside the orignial trained model file**.

* Training has been done and the next stage is **evaluation**. Actually, `FlagEmbedding` directory contains very good documentation for evaluating the fine-tuned model. I have to perform actions according to ``Custom Dataset`` heading on `FlagEmbedding.examples.evaluation`. 

* **According to the documentation I have to create an evaluation dataset for evaluating the model. This is the most important process. I have to do it ASAP**.

## Notes 04.07.2025

* I have finetuned 2 models, the first one was just trial and the second one was the main finetuned model.

* I had to evaluate them according to **custom dataset** depicted on documentation, but I have found benchmark for evaluating only **retrieval** process, and I evaluated with this benchmark (`RAG-Retrieval-Benchmark-Azerbaijani`).

* The results are not so good, model shows approximately **60%** average accuracy on datasets for this benchmark.

* As because I have finetuned *only embedder* model, I think I have to evaluate the model performance according to the documentation depicted on FlagEmbedding repo. The difficult part of this is that I have to create a dataset for evaluating the model performance. And I think my train data which contains about 320 queries are not enough. I have to search it that whether I should increase the size of the data or not. 

* My initial plan: scrape very long number of documents from e-qanun => create training dataset => create evalution dataset => finetune the model => evaluate the model.

## Notes 11.07.2025

* I have modified `src/utils/scraper.py` so that:
    * You can scrape as much documents from *e-qanun.az* as you want (**max_docs**);
    * As because all documents are under `e-qanun.az/framework/`, I am just defining **start id** and it starts scrape from `e-qanun.az/framework/start_id`.

* `src/utils/dataset_preparation.py` script is created: 
    * Opens and reads all scraped documents;
    * Cleans and creates chunks from it;
    * OpenAI client generates a query according to the sample query from each chunk;
    * Each generated query (synthetic dataset example in the format of: *query - pos - neg - prompt - type*) from each chunk are combined in a single `.jsonl` file and saved in the directory;
    * Correct formatting of generated query is done automatically for neater and cleaner `.jsonl` files;
    * Try/Except blocks have been provided for unexpected errors and timeouts. 