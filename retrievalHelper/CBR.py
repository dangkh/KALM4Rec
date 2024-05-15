# import torch
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# import torch.nn as nn
# from torch.nn.init import xavier_normal_, constant_, xavier_uniform_
# import torch.optim as optim

# class CBR(object):
#     """docstring for CBR"""
#     def __init__(self, arg):
#         super(CBR, self).__init__()
#         print("*"*50)
#         print("Initializing CBR")
#         print("*"*50)
#         self.userFT, self.userFT_test, self.rest_feature, label_train, label_test, self.config = arg
#         # size depend on BERT
#         print(self.rest_feature.shape)
#         dim_users, dim_items = 384, 384
#         learning_rate = self.config["learning_rate"]
#         hidden_dim = modelConfig['hidden_dim']
#         num_epochs = modelConfig['num_epochs']
#         self.trainLB = np.asarray(label_train)
#         self.testLB = np.asarray(label_test)
#         self.dictionary = {}
#         self.model = MatrixFactorization(dim_users, dim_items, hidden_dim).to(device)

#     def train(self, userFT, userFT_test, trainLB, testLB, train_dataset, test_dataset, rest_feature):
#         print("feature shape (train / test):")
#         print(userFT.shape, userFT_test.shape)
#         train_dataset = DataCF(userFT, trainLB)
#         test_dataset = DataCF(userFT_test, testLB)



#         train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
#         rest_feature = torch.from_numpy(rest_feature).type(torch.FloatTensor)
#         rest_feature = rest_feature.to(device)
#         mp, mr, mf = 0, 0, 0
#         print(self.model)
#         optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay= 1e-4)
#         criterion = nn.BCELoss()



#     def get_pred(self):
#         pass



# # print('*'*10, 'Result' ,'*'*10, file = sourceFile)



# # # initial trainner    

# # # Training loop
# # for epoch in range(self.num_epochs):
# #     total_loss = 0.0
# #     self.model.train()
# #     for batch_idx, batchDat in enumerate(train_loader):
# #         optimizer.zero_grad()
# #         data, label = batchDat
# #         userDat = data.to(device)
# #         label = label.to(device)
# #         restDat = rest_feature
# #         prediction = self.model(userDat, restDat)
# #         loss = criterion(prediction , label)
# #         loss.backward()
# #         total_loss += loss.item() 
# #         optimizer.step()

# #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss}')

# #     if (epoch+1) % 10 == 0:
        
# #         lResults = evaluateModel(self.model, test_loader, rest_feature, gt, test_users, args.quantity, rest_Label)

# #         p, r, f = extractResult(lResults)
# #         if mean(r) > mean(mr):
# #             mp, mr, mf = p, r, f
# #         print(f'Epoch [{epoch+1}/{num_epochs}], TestLoader prec: {mean(p)}, rec: {mean(r)}, f1: {mean(f)}')
        

# # if args.export2LLMs:
# #     json_object = json.dumps(dictionary, indent=4)
# #     with open(f"{args.city}_knn2rest_CBR.json", "w") as outfile:
# #         outfile.write(json_object)

# # p, r, f = mp, mr, mf 
# # print("args:", args, file = sourceFile)
# # print(mean(p), mean(r), mean(f), file = sourceFile)
# # print('*'*10, 'End' ,'*'*10, file = sourceFile)
# # sourceFile.close()
# # print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

