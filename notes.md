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