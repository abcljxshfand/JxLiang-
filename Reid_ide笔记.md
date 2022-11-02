# Reid_ide笔记

## data_manager.py

### Market1501类

#### init()函数

```python
dataset_dir = 'Market1501'

    def __init__(self,root = 'data',**kargs):
        self.dataset_dir = osp.join(root,self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir,'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir,'query')
        self.gallery_dir = osp.join(self.dataset_dir,'bounding_box_test')

        self._check_before_run()

        train,num_train_pids,num_train_imgs = self._process_dir(self.train_dir,relabel= True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> Market1501 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery
```



#### check()函数

```python
def _check_before_run(self):
    if not osp.exists(self.dataset_dir) :
        raise RuntimeError('{} is not available'.format(self.dataset_dir))
    if not osp.exists(self.train_dir) :
        raise RuntimeError('{} is not available'.format(self.train_dir))
    if not osp.exists(self.query_dir) :
        raise RuntimeError('{} is not available'.format(self.query_dir))
    if not osp.exists(self.gallery_dir) :
        raise RuntimeError('{} is not available'.format(self.gallery_dir))
```

#### process()函数

```python
def _process_dir(self,dir_path,relabel = False):
    img_paths = glob.glob(osp.join(dir_path,'*.jpg'))  ## 0002_c1s1_000451_03.jpg
    '''
    re.compile 预编译正则表达式，以后能直接使用。
    正则表达式：匹配字符串
    r''表示字符串形式，防止字符转义的
    ([-\d])+_c(\d):匹配字符串 xxxx（多个数字）_cx中的xxxx和x
    '''
    pattern = re.compile(r'([-\d]+)_c(\d)')

    pid_container = set()
    for img_path in img_paths :
        pid , _ = map(int,pattern.search(img_path).groups())
        if pid == -1 : continue
        pid_container.add(pid)

    pid2lable = { pid : label for label,pid in enumerate(pid_container)}

    dataset = []
    for img_path in img_paths :
        pid , cid = map(int,pattern.search(img_path).groups())
        if pid == -1 :continue
        assert 0 <=  pid <= 1501 ##assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常
        assert 1 <= cid <= 6 ## camera_id = -1 垃圾数据
        cid -= 1
        if relabel : pid = pid2lable[pid]
        dataset.append((img_path, pid, cid))

    num_pids = len(pid_container)
    num_imgs = len(dataset)
    return dataset,num_pids,num_imgs
```



## dataset_loader.py

### 编写read_img（）函数

思路：考虑两个问题，没有东西读和读不到。没有东西读的解决办法是，用os模块判断。解决读不到的办法是，循环一直读。

```python
def read_img(img_path) :
    if  not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    got_img = False

    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass

    return  img
```



### 编写ImageDataloader类

思路：继承Dataset，重构里面的一些方法。init、len、getitem，为什么要重构，因为原本的父类不适用于该项目的数据集。怎么重构，先从init开始，初始化属性，所需要的参数有dataset、transform。len，返回dataset的信息。getitem，需要索p引，来获取dataset中的相关信息，然后利用获取到的img_path读图片，

```python
class ImageLoader(Dataset) :
    def __init__(self,dataset,transforms = None):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return  len(self.dataset)

    def __getitem__(self, index):
        img_path,pid,cid= self.dataset[index]
        img = read_img(img_path)
        if self.transforms is not None :
            img = self.transforms(img)
        return  img,pid,cid
```



## transform.py

一般pytorch有自带。例子：

```python
data_transforms = {
    'train' : transforms.Compose([transforms.RandomRotation(45),
                                  transforms.CenterCrop(224),
                                  transforms.RandomHorizontalFlip(p = 0.5),
                                  transforms.RandomVerticalFlip(p = 0.5),
                                  transforms.ColorJitter(brightness = ,contrast = ,saturation = ,hue = )
                                  transforms.RondomGrayscale(p = 0.025),
                                  transforms.ToTensor()
                                  transforms.Normaliza([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    ]     
    ),
    'valid' : transforms.Compose([
        transforms.CenterCrop(224)
        transforms.ToTensor()
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    )
}

####################################################################
##使用情况：
1.直接使用 img = self.transforms(img)

2.作为参数使用 image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','valid']}
```









## ResNet.py

```python
class ResNet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        elif self.loss == {'ring'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
```



## train.py



