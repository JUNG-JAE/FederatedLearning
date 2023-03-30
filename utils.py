import sys
import os
import logging
from conf import settings

def get_network(args):
    if args.net == 'vgg16':
        from networks.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from networks.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from networks.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from networks.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from networks.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from networks.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from networks.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from networks.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from networks.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from networks.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from networks.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from networks.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from networks.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from networks.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from networks.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from networks.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from networks.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from networks.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from networks.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from networks.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from networks.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from networks.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from networks.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from networks.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from networks.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from networks.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from networks.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from networks.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from networks.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from networks.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from networks.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from networks.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from networks.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from networks.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from networks.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from networks.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from networks.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from networks.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from networks.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from networks.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from networks.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from networks.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from networks.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from networks.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def setGlobalRound(args):
    global_round = 0
    
    # set global round
    if not os.path.exists("./"+settings.LOG_DIR+"/"+args.net+"/global_model/G0"):
        print("[ ========== Global Round: 0 ========== ]")
        try:
            os.makedirs("./"+settings.LOG_DIR+"/"+args.net+"/global_model/G0")
        except OSError:
            print ('Error: Creating global model directory')
    else:
        rounds = os.listdir("./"+settings.LOG_DIR+"/"+args.net+"/global_model")
        rounds = [int(round.strip("G")) for round in rounds if round.startswith("G")]
        try:
            current_round = max(rounds)
            os.makedirs("./"+settings.LOG_DIR+"/"+args.net+"/global_model/G"+str(current_round+1))
        except OSError:
            print ('Error: Creating global model {0} directory'.format(current_round+1))

        global_round = current_round + 1
        print("[ ========== Global Round: {0} ========== ]".format(global_round))

    return global_round


def setLogger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')

    file_handler = logging.FileHandler(settings.LOG_DIR+"/"+args.net+"/FL.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger