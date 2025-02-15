# Graph-CRISPR


This is a detailed document explaining how to train and test using your own dataset. 
First, let's introduce the necessary Python files.

**datasetXX.py**  
Files named in this format represent scripts that convert raw base sequence data into graph data.  
Since our model requires graph data as input, you need to create your own graph data. Start by creating an empty folder, which can be named after your dataset. Add the folder directory to the specified parameter in the config file.  
You can copy any existing `datasetxx.py` file and make the following changes:  
1. Update the config path to your own directory:  
   ```python  
   fig = config('/data/data2/liuxiuqin/JYJ/pygall/gat/config_BE.json')  
   ```  
2. Adjust the base sequence length to the correct value:  
   ```python  
   positional_encoding(max_len=20, d_model=640), dtype=torch.float32)  
   ```  
3. Change the `.pt` file name to your dataset name:  
   ```python  
   def processed_file_names(self):  
       return ['wideRNA.pt']  
   ```  
4. Input your dataset path and the corresponding label column name:  
   ```python  
   df = pd.read_csv("/data/data2/liuxiuqin/JYJ/dataset/update_data/BE/BE_1210RNA.csv", nrows=config.length)  
   g_label = row['Efficiencies(%)']  
   ```  
After completing these settings, run your dataset script. Check if a folder named `processed` is generated in your empty folder. Inside this folder, there should be three `.pt` files: `pre_filter.pt`, `pre_transform.pt`, and your named dataset `.pt` file. If these files exist, it means the graph dataset has been successfully created.

**XX.config**  
This is the basic parameter configuration file. The following parameters can be customized:  
- `"root": "/data/data2/liuxiuqin/JYJ/dataset/update_data/BE/BE/"` ---- Path to the graph data.  
- `"rna_strc_path": "/data/data2/liuxiuqin/JYJ/dataset/update_data/BE/cleaned_be_sec.txt"` ---- Path to the sgRNA secondary structure file. Use mxfold2 to generate the secondary structure, and convert it to the corresponding `.txt` format.  
- `"model_path": "/data/data2/liuxiuqin/JYJ/pygall/gat/adjust_model/bestmodelBE"` ---- A new file to store the best model.  
- `"embed_path": "/data/data2/liuxiuqin/JYJ/dataset/update_data/BE/BE_embeddings_tensors.pkl"` ---- Path to the sgRNA embedding file, generated using RNA-FM.  
- `"length": 1133` ---- Number of input samples, corresponding to the number of sgRNAs.  
- `"epochs": 100` ---- Number of training epochs, which can be set as needed.

**trainandtestXX.py**  
This is the training script. The number of epochs can be set in the config file.  
In the main function, pay attention to the following parts:  
- `config_file_path = '/data/data2/liuxiuqin/JYJ/pygall/gat/config_BE.json'` ---- Update to your own JSON file path.  
- `best_model_dir = '/data/data2/liuxiuqin/JYJ/pygall/gat/adjust_model/bestmodelBE'` ---- Update to your own best model file path to record the best model during training.  
- `excel_path_mean = '/data/data2/liuxiuqin/JYJ/pygall/gat/adjust_output/6fold_BE85%_mean_results.xlsx'`  
- `excel_path_best = '/data/data2/liuxiuqin/JYJ/pygall/gat/adjust_output/6fold_BE85%_best_results.xlsx'` ---- Save the results of each training round. You can set your own storage path.  

In the function definition section, note the following `def` functions:  
```python  
def initialize_model(config, hidden_dim, dropout, activation_name, conv_layer, pool_layer, global_pool_layer, lr, Alpha, alpha):  
    model_path = '/data/data2/liuxiuqin/JYJ/pygall/gat/adjust_model/bestmodel_official/best_model_fold5_epoch90.pt'  
    checkpoint = torch.load(model_path)  
    model = checkpoint['best_model']  
```  
If you are using transfer learning, set `model_path` to the path of the previously best parameters. If you are training from scratch, refer to the commented section in the code to reinitialize the model.  
By running `trainandtestXX.py`, you can train your generated graph dataset. After training, you can check the saved best model in your `bestmodel` directory.

**predictXX.py**  
This is the prediction script, used to call the best model for testing on the test set.  
In the main function, pay attention to the following parts:  
- `config_file_path = '/data/data2/liuxiuqin/JYJ/pygall/gat/config_BE.json'` ---- Update to your own config file.  
- `model_path = '/data/data2/liuxiuqin/JYJ/pygall/gat/adjust_model/bestmodelBE/best_model_fold5_epoch33.pt'` ---- Update to your own best model.  
- `excel_path = '/data/data2/liuxiuqin/JYJ/pygall/gat/adjust_output/prediction_BE15%test_results.xlsx'` ---- Update to your own save path.

**adjust_model.py, adjust_layers.py, and adjust_common.py**  
These are our model framework codes. If you are not using Optuna to adjust the model framework, you can directly reference them without modification.

**npy2tensor.py**  
After using RNA-FM to obtain the embedding representation of each sgRNA, RNA-FM will generate a `representation` folder containing multiple `.npy` files, representing the embedding matrix of each sgRNA. Run this script (remember to modify the corresponding input and output paths) to generate a `.pkl` file that integrates all matrices and converts them into tensor form, which can be directly used as the `embed_path` in the config file.

**delete.py**  
After using mxfold2 to obtain the secondary structure of each sgRNA, run this script to delete redundant parts of the generated result file, producing a `.txt` file that can be directly used as the `rna_strc_path` in the config file.

The uploaded datasets and training/prediction code are provided for reference and can be modified according to your specific needs.
