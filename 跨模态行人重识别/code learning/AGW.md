# AGW代码学习

# 阅读代码的思路

## 自上而下构建项目程序的系统架构

阅读代码很容易陷入一个误区：刚开始就企图把所遇到的每一行程序都搞明白其功能和实现。**一叶障目，不见泰山**。过于纠结具体的实现细节会在最开始就把我们陷于泥潭之中而不能前进，浪费时间并消磨我们阅读代码的耐心和信心。

相反，在最开始时我们应该专注于掌握[项目程序](https://www.zhihu.com/search?q=项目程序&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1263867097})的系统架构，根据其<font color='red'>**模块儿划分和调用逻辑**</font>在脑海里构建其骨架，即要知道**<font color='red'>它先干嘛、再干嘛</font>**，为了干这个需要与哪些**<font color='red'>模块儿进行什么交互</font>**，从宏观上对项目有一个全面的认识。为了弄清其架构，我们可以采取top-down的方法，一般可以从其主函数出发，一般是main()函数或者其他函数，然后顺着项目程序的主要功能这个线索一路走下去，根据函数的名字以及层次关系确定每一个函数的大致用途即可，将理解作为注解写在这些函数的边上，不用关注这些函数的具体实现。

在阅读时，要利用uml建立各类对象之间的关系，并根据自己要展开的层级，随手在笔记本上绘制出树状结构，忽略对系统架构不重要的细节，逐渐建立该项目代码的系统架构。注意在该步骤中一定要注意展开的层级，这涉及到这部分工作量的问题，究竟需要展开到那一层，需要根据我们的系统架构定位有关，要看我们所希望的系统架构具体到哪一个层次。

## 建立系统架构和[功能逻辑](https://www.zhihu.com/search?q=功能逻辑&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1263867097})之间的关联

在完成系统架构的搭建之后，我们需要根据程序功能，理解系统架构中每一部分的功能，以及其和其他部分的逻辑交互。这一步对于我们进一步理解系统架构的逻辑，以及后面对每个功能模块儿详细功能的深入探究是很重要的。这个过程与上一个过程有重叠的部分，在对应完每部分的功能之后，我们要对整个系统功能进行模拟[逻辑推理](https://www.zhihu.com/search?q=逻辑推理&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1263867097})，确定我们猜测的各模块儿的功能是合理的。

## 核心代码重点剖析与注释

完成系统架构搭建之后，需要转入bottom-up这一部分是我们理解程序某一个程序功能的重要过程，使我们阅读代码的核心之一。此时我们应该要逐行进行阅读，搞清楚每个语句的功能和实现方法，搞清变量的意义和关联关系，搞清实现的逻辑和算法。从我们最感兴趣的一个点，开始设置断点，跟进去看发生了哪些事情，这一块儿和架构设计哪一块是match的。然后就是阅读过程中注释的书写了，这是加深程序理解的重要方法。

# train.py(AGW)

![image-20230201094908469](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20230201094908469.png)

## Start

AGW模型的**相关参数**

把它们进行简单分类：

**1.超参数：**optim、lr、epoch、batch_sizes、

**2.训练：**arch、dataset、model_path（还有其他类型的path，保留下来）

（**测试：**）

**3.设备：**gpu、worker

------

worker：与dataloader有关系。dataloader要从RAM中找到每次迭代所需要的batch，如果找到了，就使用；找不到，就吩咐worker把batch加载到内存RAM中。

worker，普通的工作进程。worker的效率，与cpu有关。

worker的数目越多，dataloader的速度越快；但是内存开销更大、cpu负担更重。

特殊情况，worker=0，工作速度慢。

------

4.行人重识别相关：margin、mode（SYSU--MM01室内、全局模式）、

```python
parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='agw', type=str,
                    metavar='m', help='method type: base or agw')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
```



##  **Data loading code**

这段代码的作用：

加载数据，对数据进行预处理。

```python
print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
```



## Preparing model

代码主体架构：

![image-20230201093343185](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20230201093343185.png)

代码：

```python
print('==> Building model..')
if args.method =='base':
    net = embed_net(n_class, no_local= 'off', gm_pool =  'off', arch=args.arch)
else:
    net = embed_net(n_class, no_local= 'on', gm_pool = 'on', arch=args.arch)
net.to(device)  ## 这代表将模型加载到指定设备上
cudnn.benchmark = True  ## 为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion_id = nn.CrossEntropyLoss()
if args.method == 'agw':
    criterion_tri = TripletLoss_WRT()
else:
    loader_batch = args.batch_size * args.num_pos
    criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)

criterion_id.to(device)
criterion_tri.to(device)


if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr
```

作用：先选择此次训练/测试时用的方法（AGW/base/其他方法）,根据之前的model.py文件来创建模型对象。在构建好模型后，需要把模型放到实验设备上运行（N个gpu/cpu）。接下来定义损失函数、优化器、学习率衰退器等模型优化工具（具体细节稍后分析）。



## Train

### train（epoch）

函数的主要架构：

![image-20230201100646622](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20230201100646622.png)

代码：

```python
def train(epoch):

    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):

        labels = torch.cat((label1, label2), 0)

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())

        labels = Variable(labels.cuda())
        data_time.update(time.time() - end)


        feat, out0, = net(input1, input2)

        loss_id = criterion_id(out0, labels)
        loss_tri, batch_acc = criterion_tri(feat, labels)
        correct += (batch_acc / 2)
        _, predicted = out0.max(1)
        correct += (predicted.eq(labels).sum().item() / 2)

        loss = loss_id + loss_tri
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.3f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss))

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)
```

作用：开始，定义（怎么定义？）相关变量，包括：各种loss、时间花销（作用？）。然后，进行循环迭代。迭代过程先从dataloader中取出所需数据，把数据放至实验设备中，在调整完模型模式后，进行实验。实验的基本流程是把数据通过模型进行前向传播（How？）、反向传播（How？）、优化器优化相关参数（How？）。最后，记录实验结果。

