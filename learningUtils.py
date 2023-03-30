import torch
import torch.nn as nn
from collections import OrderedDict

from utils import get_network
from conf import settings
from dataLoader import sourceDataLoader

# def aggregation(args, *models):
#     aggregation_model = get_network(args)
#     ordered_dic = OrderedDict()

#     for index, model in enumerate(models):
#         layers = [weights for weights in model.state_dict()]
        
#         if index == 0:
#             for layer in layers:
#                 weights = model.state_dict()[layer]
#                 ordered_dic[layer] = weights
#         elif len(models) == index + 1:
#             for layer in layers:
#                 weights = model.state_dict()[layer]
#                 ordered_dic[layer] += weights
#             for layer in layers:
#                 ordered_dic[layer] = ordered_dic[layer] / len(models)
#         else:
#             for layer in layers:
#                 weights = model.state_dict()[layer]
#                 ordered_dic[layer] += weights

#     aggregation_model.load_state_dict(ordered_dic)
    
#     return aggregation_model

def aggregation(args, *models):
    aggregation_model = get_network(args)
    aggregation_model_dict = OrderedDict()

    for index, model in enumerate(models):
        for layer in model.state_dict().keys():
            if index == 0:
                aggregation_model_dict[layer] = 1/len(models) * model.state_dict()[layer]
            else:
                aggregation_model_dict[layer] += 1/len(models) * model.state_dict()[layer]
    
    aggregation_model.load_state_dict(aggregation_model_dict)

    return aggregation_model


# Evaluate model
@torch.no_grad()
def sourceEvaluate(args, writer, model, global_round, tb=True):
    _, test_loader = sourceDataLoader()

    loss_function = nn.CrossEntropyLoss()

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()

        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Evaluating Network.....')
    print('Accuracy {:.4f}, Average loss: {:.4f}'.format(correct.float() * 100/ len(test_loader.dataset), test_loss / len(test_loader.dataset)))
    print('-----------------------------------')
    for i in range(10):
        print('Accuracy of %3s : %2d %%' % (settings.LABELS[i], 100*class_correct[i]/class_total[i]))

    # add informations to tensorboard
    if tb:
        writer.add_scalar('Aggregate Test/Average loss', test_loss / len(test_loader.dataset), global_round)
        writer.add_scalar('Aggregate Test/Accuracy', correct.float() * 100 / len(test_loader.dataset), global_round)

    return correct.float() * 100 / len(test_loader.dataset) 


# Predcit model uncertainity
def predictUncertainity(model, data, n_iter=100):
    output_list = []

    with torch.no_grad():
        for i in range(n_iter):
            output = model(data)
            output_list.append(output)
    output_mean = torch.mean(torch.stack(output_list), dim=0)
    output_std = torch.std(torch.stack(output_list), dim=0)

    return output_mean, output_std


# Averaging uncertainity
def sourceAvgUncertainty(args, model, test_loader):
    images, labels = next(iter(test_loader))

    if args.gpu:
        images = images.cuda()
        labels = labels.cuda()

    mean, std = predictUncertainity(model, images)
    mean_list, std_list = mean.tolist(), std.tolist()

    mean_avg = [round(sum([mean_list[column][row] for column in range(len(mean_list[0]))])/len(mean_list), 4) for row in range(len(mean_list[0]))]
    std_avg = [round(sum([std_list[column][row] for column in range(len(std_list[0]))])/len(std_list), 4) for row in range(len(std_list[0]))]
    
    print("Source Uncertainity \n   {0}".format(std_avg))

    return mean_avg, std_avg
