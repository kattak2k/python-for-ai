

## Setup RAG Basic pipeline with LlamaIndex

# Ingestion
  Documents -> chunks -> Embeddings -> Index

# Retrieval                   | # Synthesis     |
    Query -> Index -> Top K -> | LLm -> Response |
    Query -------------------->|                 |

![alt text](image.png)

# Feedback functions
![alt text](image-1.png)

# Context Relevance
![context arch](image-2.png)
![scores](image-3.png) --> relevance scores
![Structure](image-4.png)


### senter window retrieval

![alt text](image-5.png)

### Auto-merginv retrieval
![alt text](image-6.png)

### All different types of evaluations
![alt text](image-7.png)