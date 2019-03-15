Data processing: 
- for r252 dataset, call data_processing.py r252_build_dataset("r252")
- for hstone dataset, call hstone_process.py hstone_build_dataset("hstone")

Training 
- arguments --fourthofdata, --halfdata, --threefourthsofdata specify whether to run the model with 25%, 50% or 75% of the data
- argument --resume_ret specifies whether a retriever model has already been saved and loads its weights
- argument --dataset_name specifies which dataset to run the model with. Implemented 'r252', 'hstone'
