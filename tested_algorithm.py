# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:32:39 2021

@author: bfeng1
"""
# #%%
# train_X = []
# train_y = []
# test_X = []
# test_y = []
# for i in range(len(train_sequences)):
#     train_X.append(train_sequences[i][0])
#     train_y.append(train_sequences[i][1])
# for i in range(len(test_sequences)):
#     test_X.append(test_sequences[i][0])
#     test_y.append(test_sequences[i][1])
# #%%
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import embeddings
# from tensorflow.keras.preprocessing import sequence
# #%%
# from keras.layers.embeddings import Embedding
# #%%
# model = Sequential()
# model.add(LSTM(256, input_shape=(135, 7)))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()
# # #%%
# # from keras.preprocessing import sequence
# # import tensorflow as tf
# # from keras.models import Sequential
# # from keras.layers import Dense
# # from keras.layers import LSTM

# # from keras.optimizers import Adam
# # from keras.models import load_model
# # from keras.callbacks import ModelCheckpoint

# # #%%# build time series classification model
# # model = Sequential()
# # model.add(LSTM(256, input_shape=(135, 4)))
# # #%%
# # model.add(Dense(1, activation='sigmoid'))

# # model.summary()
# #%%
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint
# adam = Adam(lr=0.001)
# chk = ModelCheckpoint('best_model.pkl', monitor='val_acc', save_best_only=True, mode='max', verbose=1)
# model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
# model.fit(train_sequences, epochs=200, batch_size=128, callbacks=[chk], validation_data=test_sequences)
#%%
# #%%
# import torch
# from tqdm.auto import tqdm

# import torch.autograd as autograd
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import pytorch_lightning as pl
# import seaborn as sns
# from pylab import rcParams
# from matplotlib import rc
# from matplotlib.ticker import MaxNLocator

# from multiprocessing import cpu_count
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.metrics.functional import accuracy
# from sklearn.metrics import classification_report, confusion_matrix
# #%%
# class SurfaceDataset(Dataset):
#     def __init__(self, sequences):
#         self.sequences = sequences
        
#     def __len__(self):
#         return len(self.sequences)
#     def __getitem__(self,idx):
#         sequence, label = self.sequences[idx]
#         return dict(
#             sequence = torch.Tensor(sequence.to_numpy()),
#             label = torch.tensor(label).long()
#             )
# class SurfaceDataModule(pl.LightningDataModule):
#     def __init__(self,train_sequences, test_sequences, batch_size):
#         super().__init__()
#         self.train_sequences = train_sequences
#         self.test_sequences = test_sequences
#         self.batch_size = batch_size
        
#     def setup(self, stage = None):
#         self.train_sequences = SurfaceDataset(self.train_sequences)
#         self.test_dataset = SurfaceDataset(self.test_sequences)
#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size = self.batch_size,
#             shuffle = True,
#             num_workers = cpu_count()
#         )
#     def val_dataloader(self):
#         return DataLoader(
#             self.test_dataset,
#             batch_size = self.batch_size,
#             shuffle = False,
#             num_workers = cpu_count()
#             )
#     def test_dataloader(self):
#         return DataLoader(
#             self.test_dataset,
#             batch_size = self.batch_size,
#             shuffle = False,
#             num_workers = cpu_count()
#             )
# #%%
# N_EPOCHS = 250
# BATCH_SIZE = 64

# data_module = SurfaceDataModule(train_sequences, test_sequences, BATCH_SIZE)
# #%%
# class SequenceModel(nn.Module):
#     def __init__(self, n_features, n_classes, n_hidden = 256, n_layers=3):
#         super().__init__()
#         self.n_hidden = n_hidden
#         self.lstm = nn.LSTM(
#             input_size = n_features,
#             hidden_size = n_hidden,
#             num_layers = n_layers,
#             batch_first = True,
#             dropout = 0.75)
#         self.classifer = nn.Linear(n_hidden, n_classes)
#     def forward(self, x):
#         self.lstm.flatten_parameters()
#         _,(hidden, _) = self.lstm(x)
        
#         out = hidden[-1]
#         return self.classifier(out)
        
# class SurfacePredictor(pl.LightningModule):
#     def __init__(self, n_features: int, n_classes: int):
#         super().__init__()
#         self.model = SequenceModel(n_features, n_classes)
#         self.criterion = nn.CrossEntropyLoss()
        
#     def forward(self, x, labels = None):
#         output = self.model(x)
#         loss = 0
#         if labels is not None:
#             loss = self.criterion(output, labels)
#         return loss, output
#     def training_step(self, batch, batch_idx):
#         sequences = batch['sequence']
#         labels = batch['label']
#         loss, outputs = self(sequences, labels)
#         predictions = torch.argmax(outputs, dim = 1)
#         step_accuracy = accuracy(predictions, labels)
        
#         self.log('train_loss', loss, prog_bar = True, logger = True)
#         self.log('train_accuracy', step_accuracy, prog_bar = True, logger = True)
#         return {"loss": loss,  "accuracy": step_accuracy}
    
#     def validation_step(self, batch, batch_idx):
#         sequences = batch['sequence']
#         labels = batch['label']
#         loss, outputs = self(sequences, labels)
#         predictions = torch.argmax(outputs, dim = 1)
#         step_accuracy = accuracy(predictions, labels)
        
#         self.log('val_loss', loss, prog_bar = True, logger = True)
#         self.log('val_accuracy', step_accuracy, prog_bar = True, logger = True)
#         return {"loss": loss,  "accuracy": step_accuracy}

#     def test_step(self, batch, batch_idx):
#         sequences = batch['sequence']
#         labels = batch['label']
#         loss, outputs = self(sequences, labels)
#         predictions = torch.argmax(outputs, dim = 1)
#         step_accuracy = accuracy(predictions, labels)
        
#         self.log('test_loss', loss, prog_bar = True, logger = True)
#         self.log('test_accuracy', step_accuracy, prog_bar = True, logger = True)
#         return {"loss": loss,  "accuracy": step_accuracy}

#     def configure_optimizers(self):
#         return optim.Adam(self.parameters(), lr = 0.0001)
    
# #%%
# model = SurfacePredictor(
#     n_features = len(feature_columns),
#     n_classes = len(label_encoder.classes_)
# )

# checkpoint_callback = ModelCheckpoint(
#     dirpath="checkpoints",
#     filename = "best_checkpoint",
#     save_top_k=1,
#     verbose=True,
#     monitor = "val_loss",
#     mode = "min")
# logger = TensorBoardLogger("lightning_logs", name = "surface")

# trainer = pl.Trainer(
#     logger = logger,
#     # checkpoint_callback = checkpoint_callback,
#     max_epochs = N_EPOCHS,
#     progress_bar_refresh_rate = 30)

# #%%
# trainer.fit(model, data_module)


#%%
# from tslearn.metrics import dtw
# score_list = []
# for i in range(len(good_dataset)):
#     for j in range(len(good_dataset)):
#         dtw_score = dtw(good_dataset[i]['left angle ratio'],good_dataset[j]['left angle ratio'])
#         score_list.append(dtw_score)
# print(np.mean(score_list))
# #%%
# test = inner_dataset[6]['left angle ratio']
# scores = []
# for i in range(len(good_dataset)):
#     scores.append(dtw(good_dataset[i]['left angle ratio'],test))
# print(np.mean(scores))
#%%

# #%%
# from scipy.fft import fft, rfftfreq,rfft
# test = good_dataset[17]
# plt.plot(test.index,test['left angle ratio'])

# plt.figure()
# y = test['left angle ratio'].to_list()
# x = len(y)
# fy = np.abs(rfft(test['left angle ratio'].to_list()))
# fx = rfftfreq(x)
# plt.plot(fx[2:], fy[2:])

# #%%
# from statsmodels.tsa.seasonal import seasonal_decompose
# from dateutil.parser import parse
# result_mul = seasonal_decompose(test['left angle ratio'], model='multiplicative', extrapolate_trend='freq')
# result_add = seasonal_decompose(test['left angle ratio'], model='additive', extrapolate_trend='freq')

# plt.rcParams.update({'figure.figsize': (10,10)})
# result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
# result_add.plot().suptitle('Additive Decompose', fontsize=22)
# plt.show()
#%%
# def create_pipelines(seed):
#     models = [
#             ("Nearest Neighbors",KNeighborsClassifier(3)),
#     ("Linear SVM",SVC(kernel="linear", C=0.025)),
#     ("RBF SVM",SVC(gamma=2, C=1)),
#     ("Gaussian Process",GaussianProcessClassifier(1.0 * RBF(1.0))),
#     ("Decision Tree",DecisionTreeClassifier(max_depth=5)),
#     ("Random Forest",RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
#     ("Neural Net",MLPClassifier(alpha=1, max_iter=1000)),
#     ("AdaBoost",AdaBoostClassifier()),
#     ("Naive Bayes",GaussianNB()),
#     ("QDA",QuadraticDiscriminantAnalysis())]
    
#     scalers = [('StandardScaler', StandardScaler()),
#                ('MinMaxScaler', MinMaxScaler()),
#                ('MaxAbsScaler', MaxAbsScaler()),
#                ('RobustScaler', RobustScaler()),
#                ('QuantileTransformer-Normal', QuantileTransformer(output_distribution='normal')),
#                ('QuantileTransformer-Uniform', QuantileTransformer(output_distribution='uniform')),
#                ('PowerTransformer-Yeo-Johnson', PowerTransformer(method='yeo-johnson')),
#                ('Normalizer', Normalizer())]
#     additions= [('PCA', PCA(n_components = 4))]
#     # create pipelines
#     pipelines = []
#     for model in models:
#         model_name = "_" + model[0]
#         pipelines.append(model_name, Pipeline([model]))
        
#         # append model+scaler
#         for scaler in scalers:
#             model_name = scaler[0] + "_"+ model[0]
#             pipelines.append((model_name, Pipeline([scaler, model])))
            
#         # Append model+addition
#         for addition in additions:
#             model_name = "_" + model[0] + "-" + addition[0]
#             pipelines.append((model_name, Pipeline([addition, model])))
#         # append model +scaler+addition
#         for scaler in scalers:
#             for addition in additions:
#                 model_name = scaler[0]+"_"+model[0]+"-"+addition[0]
#                 pipelines.append((model_name, Pipeline([scaler, addition, model])))
        
#         return pipelines
# def run_cv_and_test(X_train, y_train, X_test, y_test, pipelines, scoring, seed, 
#                     num_folds, dataset_name, n_jobs):
#     rows_list = []
#     #Lists for the pipeline results
#     results = []
#     names = []
#     test_scores = []
#     prev_clf_name = pipelines[0][0].split("_")[1]
#     print("first name is : ", prev_clf_name)
    
#     for name, model in pipelines:
#         kfold = model_selection.KFold(n_splits = num_folds, random_state=seed)
#         cv_results = model_selection.cross_val_score(model, X_train, y_train,
#                                                      cv = kfold,
#                                                      n_jobs = n_jobs,
#                                                      scoring = scoring)
#         results.append(cv_results)
#         names.append(name)
        
#         # print CV results of the best CV classier
#         msg = "%s: %f (%f)" % (name,cv_results.mean(),cv_results.std())
#         print(msg)
        
#         # fit on train and predict on test
#         model.fit(X_train, y_train)
#         if scoring == "accuracy":
#             curr_test_score = model.score(X_test, y_test)
#         elif scoring == "roc_auc":
#             y_pred = model.predict_proba(X_test)[:, 1]
#             curr_test_score = roc_auc_score(y_test, y_pred)

#         test_scores.append(curr_test_score)

#         # Add separation line if different classifier applied
#         rows_list, prev_clf_name = check_seperation_line(name, prev_clf_name, rows_list)

#         # Add for final dataframe
#         results_dict = {"Dataset": dataset_name,
#                         "Classifier_Name": name,
#                         "CV_mean": cv_results.mean(),
#                         "CV_std": cv_results.std(),
#                         "Test_score": curr_test_score
#                         }
#         rows_list.append(results_dict)

#     print_results(names, results, test_scores)

#     df = pd.DataFrame(rows_list)
#     return df[["Dataset", "Classifier_Name", "CV_mean", "CV_std", "Test_score"]]       
        
# def check_seperation_line(name, prev_clf_name, rows_list):
#     """
#         Add empty row if different classifier ending
#     """

#     clf_name = name.split("_")[1]
#     if prev_clf_name != clf_name:
#         empty_dict = {"Dataset": "",
#                       "Classifier_Name": "",
#                       "CV_mean": "",
#                       "CV_std": "",
#                       "Test_acc": ""
#                       }
#         rows_list.append(empty_dict)
#         prev_clf_name = clf_name
#     return rows_list, prev_clf_name

# def print_results(names, results, test_scores):
#     print()
#     print("#" * 30 + "Results" + "#" * 30)
#     counter = 0

#     class Color:
#         PURPLE = '\033[95m'
#         CYAN = '\033[96m'
#         DARKCYAN = '\033[36m'
#         BLUE = '\033[94m'
#         GREEN = '\033[92m'
#         YELLOW = '\033[93m'
#         RED = '\033[91m'
#         BOLD = '\033[1m'
#         UNDERLINE = '\033[4m'
#         END = '\033[0m'

#     # Get max row
#     clf_names = set([name.split("_")[1] for name in names])
#     max_mean = {name: 0 for name in clf_names}
#     max_mean_counter = {name: 0 for name in clf_names}
#     for name, result in zip(names, results):
#         counter += 1
#         clf_name = name.split("_")[1]
#         if result.mean() > max_mean[clf_name]:
#             max_mean_counter[clf_name] = counter
#             max_mean[clf_name] = result.mean()

#     # print max row in BOLD
#     counter = 0
#     prev_clf_name = names[0].split("_")[1]
#     for name, result, score in zip(names, results, test_scores):
#         counter += 1
#         clf_name = name.split("_")[1]
#         if prev_clf_name != clf_name:
#             print()
#             prev_clf_name = clf_name
#         msg = "%s: %f (%f) [test_score:%.3f]" % (name, result.mean(), result.std(), score)
#         if counter == max_mean_counter[clf_name]:
#             print(Color.BOLD + msg)
#         else:
#             print(Color.END + msg)   