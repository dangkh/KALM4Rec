import argparse
from ast import parse
import os
import time
from retrievalHelper.utils import *
from retrievalHelper.u4Res import *
from retrievalHelper.u4KNN import *
from retrievalHelper.u4train import *
from retrievalHelper.models import *
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")

random_seed = 1001
setSeed(random_seed)

listcity = ['charlotte', 'edinburgh', 'lasvegas', 'london', 'phoenix', 'pittsburgh', 'singapore']

parser = argparse.ArgumentParser()
parser.add_argument('--city', type=str, default='edinburgh', help=f'choose city{listcity}')
parser.add_argument('--quantity', type=int, default=20, help='number of keyword retrieval')
parser.add_argument('--seed', type=int, default=1001, help='number of keyword retrieval')
parser.add_argument('--edgeType', type=str, default='IUF', help='future work, current using TF-IDF')

'''
Export args
'''
parser.add_argument('--logResult', type=str, default='./log', help='write log result detail')
parser.add_argument('--export2LLMs', action='store_true', help='whether export list of data for LLMs or not. \
                                                                default = False')

'''
Model args
'''
parser.add_argument('--RetModel', type=str, default='MPG_old', help='Jaccard, MF, MVAE, CBR, MPG_old, MPG')
parser.add_argument('--numKW4FT', type=int, default=20, help='number of keyword for feature')

'''
CBR args
'''
parser.add_argument('--hidden_dim', type=int, default=32, help='hidden_dim')
parser.add_argument('--lr', type=float, default=0.03, help='learning_rate')
parser.add_argument('--num_epochs', type=int, default=500, help='num_epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')

args = parser.parse_args()

print("args:", args)
print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print("Loading training keyword")
trainDat, testDat = data_reviewLoader(args.city)
train_users, train_users2kw = extract_users(trainDat['np2users'])
test_users, test_users2kw = extract_users(testDat['np2users'])

# extract user2rest for label
gt = load_groundTruth(f'./data/reviews/{args.city}.csv')
# load edgeType


keywordScore, keywordFrequence = load_kwScore(args.city, args.edgeType)
restGraph = retaurantReviewG([trainDat, keywordScore, keywordFrequence, \
                                args.quantity,  args.edgeType, gt])

KNN = neighbor4kw(f'{args.city}_kwSenEB_pad', testDat,  restGraph)
rest_Label = getRestLB(trainDat['np2rests'])
sourceFile = open(args.logResult, 'a')


print('*'*10, 'Result' ,'*'*10, file = sourceFile)
prediction = []

if args.RetModel == "CBR":
    print("*"*50)
    print("Using CBR")
    print("*"*50)

    userFT, userFT_test, rest_feature = KNN.loadFT(args.numKW4FT, rest_Label, args.city)
    label_train, label_test = label_ftColab(train_users, test_users, gt, restGraph.numRest, rest_Label)


    dim_users, dim_items = 384, 384
    learning_rate = args.lr
    hidden_dim = args.hidden_dim
    num_epochs = args.num_epochs

    trainLB = np.asarray(label_train)
    testLB = np.asarray(label_test)

    print("feature shape (train / test):")
    print(userFT.shape, userFT_test.shape)
    train_dataset = DataCF(userFT, trainLB)
    test_dataset = DataCF(userFT_test, testLB)
    model = MatrixFactorization(dim_users, dim_items, hidden_dim).to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    rest_feature = torch.from_numpy(rest_feature).type(torch.FloatTensor)
    rest_feature = rest_feature.to(device)

    # Model, optimizer, and loss function

    mp, mr, mf = 0, 0, 0
    print(model)
    # stop
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay= 1e-4)
    criterion = nn.BCELoss()
    criterion2 = nn.MSELoss()
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

            # Backward pass and optimization step
            loss.backward()
            total_loss += loss.item() 
            optimizer.step()

            # Print training progress
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss}')

        if (epoch+1) % 10 == 0:
            
            lResults = evaluateModel(model, test_loader, rest_feature, gt, test_users, args.quantity, rest_Label)
            p, r, f = extractResult(lResults)
            if mean(r) > mean(mr):
                mp, mr, mf = p, r, f
            print(f'Epoch [{epoch+1}/{num_epochs}], prec: {mean(p)}, rec: {mean(r)}, f1: {mean(f)}')
elif args.RetModel == "Jaccard":
    print("*"*50)
    print("using Jaccard")
    print("*"*50)
    jcsim = JaccardSim(f'./data/score/{args.city}-keywords-TFIUF.json', args.quantity)
    lResults = []
    for idx in tqdm(range(len(test_users))):
        testUser = test_users[idx]
        testkey = test_users2kw[idx]
        pred = jcsim.pred(testkey)
        groundtruth = gt[testUser]
        prediction.append(pred)
        score = quick_eval(pred, groundtruth, sourceFile)
        lResults.append(score)
    mp, mr, mf = extractResult(lResults)
elif args.RetModel == "MVAE":
    pass
elif args.RetModel == "MF":
    print("*"*50)
    print("running MF-BPR at: ")
    print("*"*50)

    numUser = len(train_users)
    numItem = len(rest_Label)

    label_train, label_test = label_ftColab(train_users, test_users, gt, restGraph.numRest, rest_Label)

    learning_rate = args.lr
    embedding_dim = args.hidden_dim
    num_epochs = args.num_epochs

    trainLB = np.asarray(label_train)
    testLB = np.asarray(label_test)
    train_dataset = DataBPR(trainLB)
    test_dataset = DataBPR(trainLB, True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = MFBPR(numUser, numItem, embedding_dim).to(device)

    mp, mr, mf = 0, 0, 0
    print(model)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay= 1e-4)
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()
        for batch_idx, batchDat in enumerate(train_loader):
            uid, pid, nid = batchDat
            uid = uid.to(device)
            pid = pid.to(device)
            nid = nid.to(device)

            optimizer.zero_grad()
            pred_i, pred_j = model(uid, pid, nid)
            loss = - ((pred_i - pred_j).sigmoid()+1e-10).log().sum()
            loss.backward()
            total_loss += loss.item() 
            optimizer.step()

            # Print training progress
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss}')

        if (epoch+1) % 10 == 0:
            
            lResults = evaluateModel_MFBPR(model, test_loader, "", gt, train_users, args.quantity, rest_Label)
            p, r, f = extractResult(lResults)
            if mean(r) > mean(mr):
                mp, mr, mf = p, r, f
            print(f'Epoch [{epoch+1}/{num_epochs}], prec: {mean(p)}, rec: {mean(r)}, f1: {mean(f)}')
    
    lResults = []
    for idx in tqdm(range(len(test_users))):
        if idx > 3000:
            break
        testUser, topK_Key, topUser = procesTest(test_users, test_users2kw, idx, KNN, restGraph, True)
        idU = torch.LongTensor([train_users.index(x) for x in topUser]).to(device)
        rest_score = []
        for rest in range(restGraph.numRest):
            restID = torch.LongTensor([rest]).to(device)
            rest_score.append(model.csPrediction(idU,restID))
        
        tmp = np.argsort(rest_score)[::-1][:args.quantity]
        restPred = [rest_Label[x] for x in tmp]

        groundtruth = gt[testUser]
        score = quick_eval(restPred, groundtruth)
        lResults.append(score)
    mp, mr, mf = extractResult(lResults)

elif args.RetModel == "MPG_old":
    pass
else:
    print("*"*50)
    print("using MPG")
    print("*"*50)
    l_rest = restGraph.listRestCode
    kw_data = KNN.kw_data
    lu, li, lh = list([]), list([]), list([])
    for rest in tqdm(keywordScore):
        kw_scs = keywordScore[rest]
        u = l_rest.index(rest)
        tmpI, tmpH = [], []
        for kw, sc in kw_scs:
            tmpI.append(kw_data.index(kw))
            tmpH.append(sc)
        tmpU = [u] * len(tmpI)
        lu.extend(tmpU)
        li.extend(tmpI)
        lh.extend(tmpH)

    adj = np.zeros([len(l_rest), len(kw_data)])
    for ii in range(len(lu)):
        u, v, w = lu[ii], li[ii], lh[ii]
        if args.edgeType == "IUF":
            adj[u, v] = w
        else:
            adj[u, v] = 1

    lResults = []
    lidx = [x for x in range(len(test_users))]
    np.random.shuffle(lidx)
    for ite in tqdm(range(len(test_users))):
        idx = lidx[ite]
        testUser, topK_Key = procesTest(test_users, test_users2kw, idx, KNN, restGraph)
        testkey = [kw_data.index(x) for x in topK_Key]
        ft = np.zeros(len(kw_data))
        for x in testkey: ft[x] = 1.0
        ft = ft.reshape(-1, 1)
        tmp = np.matmul(adj, ft).reshape(-1)
        idxrest = np.argsort(tmp)[::-1]
        result = [l_rest[x] for x in idxrest[:args.quantity]]
        prediction.append(result)
        groundtruth = gt[testUser]
        score = quick_eval(result, groundtruth)
        lResults.append(score)
    mp, mr, mf = extractResult(lResults)


if args.export2LLMs:
    if args.RetModel == "CBR":
        prediction = evaluateModel(model, test_loader, rest_feature, gt, test_users, args.quantity, rest_Label, True)
    json_object = json.dumps(prediction, indent=4)
    with open(f"{args.city}_pred_CBR.json", "w") as outfile:
        outfile.write(json_object)

p, r, f = mp, mr, mf 
print("args:", args, file = sourceFile)
print(mean(p), mean(r), mean(f), file = sourceFile)
print('*'*10, 'End' ,'*'*10, file = sourceFile)
sourceFile.close()
print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))


















