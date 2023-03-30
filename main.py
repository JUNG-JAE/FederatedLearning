# ----------- Learning library ----------- #
import torch
from torch.utils.tensorboard import SummaryWriter

# ------------ system library ------------ #
import os
import argparse
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ------------ custom library ------------ #
from conf import settings
from worker import Worker
from utils import setGlobalRound, setLogger, get_network
from learningUtils import aggregation, sourceEvaluate, sourceAvgUncertainty
from dataLoader import sourceDataLoader
# ---------------------------------------- #


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    args = parser.parse_args()
    
    writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW))
    logger = setLogger(args)
    
    for _ in range(settings.TOTAL_ROUND):
        
        global_round = setGlobalRound(args)

        logger.info("[ ========== Global Round: {0} ========== ]".format(global_round))

        workers = [Worker('worker'+str(worker_index), global_round, args, writer) for worker_index in range(settings.WORKER_NUM)]
        print("[ Number of worker: {0} ]".format(len(workers)))

        model_list = []
    
        for worker in workers:
            print("<--------- {0} --------->".format(worker.worker_id))
            worker.train()
            worker.evaluate()
            model_list.append(worker.model)
            _, avg_std = worker.avgUncertainty()
            logger.info('{0} Uncertainity {1}'.format(worker.worker_id, avg_std))
            
            _, test_loader = sourceDataLoader()
            _, aggregation_std = sourceAvgUncertainty(args, worker.model, test_loader)
            
            print(" ")

        print("[ Aggregation Model ]")
        aggregation_model = aggregation(args, *model_list)
        
        _, test_loader = sourceDataLoader()
        sourceEvaluate(args, writer, aggregation_model, global_round)

        torch.save(aggregation_model.state_dict(), settings.LOG_DIR+"/"+args.net+"/global_model/G"+str(global_round)+"/aggregation.pt")

        _, aggregation_std = sourceAvgUncertainty(args, aggregation_model, test_loader)
        logger.info('Aggregate Uncertainity {0}'.format(aggregation_std))

    writer.close()
