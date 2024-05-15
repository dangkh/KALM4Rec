import world
import os
import time
from retrievalHelper.utils import *
from retrievalHelper.u4Res import *
from retrievalHelper.u4KNN import *
from retrievalHelper.u4train import *
from retrievalHelper.models import *
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")

# from retrievalHelper.CBR import CBR


print("Running retrieval... ")
print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print("INFO: ", world.config)

city = world.config['city']
print("Loading training keyword")
trainDat, testDat = data_reviewLoader(city)

train_users, train_users2kw = extract_users(trainDat['np2users'])
test_users, test_users2kw = extract_users(testDat['np2users'])

# extract user2rest for label
groundtruth = load_groundTruth(f'./data/reviews/{city}.csv')

keywordScore, keywordFrequence = load_kwScore(city, world.config["edgeType"])


restGraph = retaurantReviewG([trainDat, keywordScore, keywordFrequence, \
                                world.config["quantity"], world.config["edgeType"], groundtruth])

KNN = neighbor4kw(f'{city}_kwSenEB_pad', testDat,  restGraph)
rest_Label = getRestLB(trainDat['np2rests'])

sourceFile = open(world.config["logResult"], 'a')
print('*'*10, 'Result' ,'*'*10, file = sourceFile)
mp, mr, mf = 0, 0, 0
dictionary = {}
if world.config["RetModel"] == "CBR":
	userFT, userFT_test, rest_feature = KNN.loadFT(world.config["numKW4FT"], rest_Label, world.config["city"])
	label_train, label_test = label_ftColab(train_users, test_users, groundtruth, restGraph.numRest, rest_Label)
	dim_users, dim_items = 384, 384
    learning_rate = world.modelConfig["learning_rate"]
    hidden_dim = world.modelConfig['hidden_dim']
    num_epochs = world.modelConfig['num_epochs']

	trainLB = np.asarray(label_train)
	testLB = np.asarray(label_test)
	print("feature shape (train / test):")
    print(userFT.shape, userFT_test.shape)
    train_dataset = DataCF(userFT, trainLB)
    test_dataset = DataCF(userFT_test, testLB)
    model = MatrixFactorization(dim_users, dim_items, hidden_dim).to(device)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
	rest_feature = torch.from_numpy(rest_feature).type(torch.FloatTensor)
	rest_feature = rest_feature.to(device)

	
	print(model)
	optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay= 1e-4)
	criterion = nn.BCELoss()
	# Training loop
	for epoch in range(num_epochs):
	    total_loss = 0.0
	    model.train()
	    for batch_idx, batchDat in enumerate(train_loader):
	        optimizer.zero_grad()
            data, label = batchDat
            userDat = data.to(device)
            label = label.to(device)
            restDat = rest_feature
            prediction = model(userDat, restDat)
            loss = criterion(prediction , label)
        loss.backward()
        total_loss += loss.item() 
        optimizer.step()

    	print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss}')

    	if (epoch+1) % 10 == 0:
        
	        lResults = evaluateModel(model, test_loader, rest_feature, groundtruth, test_users, world.config['quantity'], rest_Label)

	        p, r, f = extractResult(lResults)
	        if mean(r) > mean(mr):
	            mp, mr, mf = p, r, f
	        print(f'Epoch [{epoch+1}/{num_epochs}], prec: {mean(p)}, rec: {mean(r)}, f1: {mean(f)}')
	        

if world.config["export2LLMs"]:
    json_object = json.dumps(dictionary, indent=4)
    with open(f"{city}_knn2rest_{world.config['RetModel']}.json", "w") as outfile:
        outfile.write(json_object)

p, r, f = mp, mr, mf 
print("args:", world.config, file = sourceFile)
print(mean(p), mean(r), mean(f), file = sourceFile)
print('*'*10, 'End' ,'*'*10, file = sourceFile)
sourceFile.close()
print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))


