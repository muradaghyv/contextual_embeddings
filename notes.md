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