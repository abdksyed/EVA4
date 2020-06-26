import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

from .load_data import *
from .models import *
from .plotting import *
from .training import *

class Sara():
    def __init__(self, device='cuda', model_name= CusResNet18, criterion= nn.CrossEntropyLoss(), 
                    optimizer= 'SGD'):

        self.device = device
        self.model_name = model_name().to(self.device)
        self.criterion = criterion
        self.optimizer = getattr(torch.optim, optimizer)
        self.scheduler = None
        self.best_lr = None
        self.lr_variations = []
        self.best_model = None

    def loaders(self, trans=CIFAR10_AlbumTrans, aug= None, dataset= CIFAR10DataLoader, batch_size= 128, split=None):
        transformation = trans(augs= aug)
        self.split = split

        if self.split:
            self.data = dataset(transformation, batch_size, split= self.split)
        else:
            self.data = dataset(transformation, batch_size)

        return self.data.get_loaders() 

    def display_samples(self, loader, n, dataset_used):
        return display(loader, n, dataset_used)

    def model(self, input_size= (3,32,23)):
        return self.model_name, model_summary(self.model_name, input_size=input_size)
    
    def find_lr(self, loader, min_lr=1e-7, end_lr=100, num_iter= 100, momentum= 0.9, weight_decay= 0,
                nesterov= False, step_mode= 'exp', log_lr= True, best_lr_from = 'loss'):
        optimizer = self.optimizer(self.model_name.parameters(), lr=min_lr, momentum= momentum,
                                        weight_decay= weight_decay, nesterov= nesterov)
        self.lr_finder = LRFinder(self.model_name, optimizer, self.criterion, self.device)
        self.lr_finder.range_test(loader, end_lr=end_lr, num_iter=num_iter, step_mode= step_mode)
        if best_lr_from == 'loss'.lower(): 
            self.best_lr = self.lr_finder.lr_for_minloss
        elif best_lr_from == 'accuracy'.lower():
            self.best_lr = self.lr_finder.lr_for_maxacc
        else:
            self.best_lr = min_lr
        self.lr_finder.plot(log_lr= log_lr)
        self.lr_finder.reset()

    def compare_find_lr(self, loader, min_lr=1e-7, end_lr=100, num_iter= 100, momentum= 0.9, weight_decays= [],
                nesterov= False, step_mode= 'exp', log_lr= True, best_lr_from = 'loss'):
        lr = {}
        loss = {}
        acc = {}
        for i, weight_decay in enumerate(weight_decays):
            optimizer = self.optimizer(self.model_name.parameters(), lr=min_lr, momentum= momentum,
                                        weight_decay= weight_decay, nesterov= nesterov)
            self.lr_finder = LRFinder(self.model_name, optimizer, self.criterion, self.device)
            print('='*20 + f' For weight_decay: {weight_decay} ' + '='*20)
            self.lr_finder.range_test(loader, end_lr=end_lr, num_iter=num_iter, step_mode= step_mode)

            lr[i] = self.lr_finder.history['lr']
            loss[i]= self.lr_finder.history['loss']
            acc[i]= self.lr_finder.history['acc']
            self.lr_finder.reset()
        fig, ax = plt.subplots(1,2, figsize= (14,6))
        
        for i in range(len(weight_decays)):
            ax[0].plot(lr[i][10:-5], loss[i][10:-5], label=str(weight_decays[i]))
        ax[0].set_xscale('log')
        ax[0].legend(loc='upper left')
        ax[0].set_xlabel("Learning rate")
        ax[0].set_ylabel("Loss")

        for i in range(len(weight_decays)):
            ax[1].plot(lr[i][10:-5], acc[i][10:-5], label=str(weight_decays[i]))
        ax[1].set_xscale('log')
        ax[1].legend(loc='upper left')
        ax[1].set_xlabel("Learning rate")
        ax[1].set_ylabel("Accuracy")


    def lr_range_test(self, device, epochs, model, loader, start_lr=1e-4, end_lr=10,
                 criterion= nn.CrossEntropyLoss(), optimizer_name= 'SGD', smoothing= 0.05, best_lr_from = 'loss'):
        self.range_test = Range_Test(device=device, epochs= epochs, model_name= model, loader= loader, start_lr=start_lr, 
        end_lr=end_lr, criterion= criterion, optimizer_name= optimizer_name, smoothing= smoothing)
        self.range_test.range_test()
        self.range_test.plot()
        
        if best_lr_from == 'loss'.lower():
            self.best_lr = self.range_test.lr_loss
        elif best_lr_from == 'accuracy'.lower():
            self.best_lr = self.range_test.lr_acc
        else:
            self.best_lr = start_lr

    def fit(self, train_loader, test_loader, val_loader=None, epochs= 10, scheduler={'name':None}, lr=0.01, momentum=0.9, weight_decay= 0, plot_lr= False):
        
        self.val_loader = val_loader

        if self.best_lr is None:
            self.best_lr = lr
        
        optimizer = self.optimizer(self.model_name.parameters(), lr= self.best_lr, momentum= momentum, weight_decay= weight_decay)
        
        if scheduler['name'] is not None:
                if scheduler['name'] == 'OneCycleLR':
                    sched = getattr(lr_scheduler, scheduler['name'])
                    sched = sched(optimizer, max_lr= self.best_lr, epochs= scheduler['epochs'], 
                        steps_per_epoch= scheduler['steps'],
                        pct_start = scheduler.get('pct_start', 0.3),
                        div_factor= scheduler.get('div_factor', 25),
                        final_div_factor= scheduler.get('final_div_factor', 1e4),
                        last_epoch= scheduler.get('last_epoch', -1),
                        anneal_strategy= scheduler.get('anneal_strategy', 'cos'))

        self.train = Train(self.model_name, self.device, train_loader, optimizer, self.criterion)
        if self.val_loader:
            self.test = Test(self.model_name, self.device, val_loader, self.criterion)
        else:
            self.test = Test(self.model_name, self.device, test_loader, self.criterion)
        
        print('='*20 + 'START' + '='*20)
        for epoch in range(epochs):
            print('='*20 + f' EPOCH: {epoch+1} ' + '='*20)
            print(f'LR used: {optimizer.param_groups[0]["lr"]}')
            self.lr_variations.append(optimizer.param_groups[0]['lr'])
            self.train.train(sched)
            if scheduler['name'] is not None and scheduler['name'] != 'OneCycleLR':
                sched.step()
            self.test.test()
        
        if plot_lr:
            lr_used = set(self.lr_variations)
            print('='*50)
            print(f'The Learning Rates used: {lr_used}')
            plt.plot(self.lr_variations)
            plt.title('Change in Learnig Rate')
            plt.xlabel('epochs')
            plt.ylabel('Learning Rate')
            plt.show()

        print('='*20 + ' RESULTS ' + '='*20)
        print('Best Train Accuracy: ', max(self.train.train_endacc))
        if self.val_loader:
            print('Best Validation Accuracy: ', max(self.test.test_acc))
            print('Runnign Best Model on Test Loader:')
            self.best_model = self.model_name.to(self.device)
            self.best_model.load_state_dict(torch.load('/content/classifier.pt'))
            self.best_model.eval()
            self.final_test = Test(self.model_name, self.device, test_loader, self.criterion)
            self.final_test.test()
            print('Best Testing Accuracy: ', max(self.final_test.test_acc))
        else:
            print('Best Testing Accuracy: ', max(self.test.test_acc))
        print('='*49)
        
    def plot_graphs(self, test_loader, accuracy_plot= True, testvtrain_plot= True, classwise_acc= True):
        
        try:
            self.best_model.eval()
        except:
            self.best_model = self.model_name.to(self.device)
            self.best_model.load_state_dict(torch.load('/content/classifier.pt'))
            self.best_model.eval()
        
        print('Best Model Loaded!')

        if accuracy_plot:
            acc_loss(self.train, self.test)
            plt.show()
        if testvtrain_plot:
            testvtrain(self.train, self.test)
            plt.show()
        if classwise_acc:
            class_acc(self.best_model, self.device, test_loader)
        
    def missed_images(self, test_loader, dataset_used='CIFAR10', miss_no_images=36, cam_no_images= 36, GradCAM_pred= True, GradCAM_act = True,
                        layers=['layer1', 'layer2', 'layer3', 'layer4'],
                        hm_lay=0.4, img_lay= 0.6, display_layer= None):
        self.classes = classes
        if display_layer is None:
            display_layer = len(layers)
        mis(self.best_model, self.device, test_loader, miss_no_images, dataset_used= dataset_used)
        plt.show()
        for layer in layers:
            if GradCAM_pred:
                gen_cam(self.best_model, layer, hm_lay=hm_lay, img_lay= img_lay, dataset_used= dataset_used)
            if GradCAM_act:
                gen_cam(self.best_model, layer, class_idx = true_list, 
                                hm_lay=hm_lay, img_lay= img_lay, dataset_used= dataset_used)
        
        print('*'*10 + 'Grad-CAM of Mis Classified Images with respect to Predicted(wrong) Class' + '*'*10)
        plot_pred_cam(cam_no_images, display_layer)
        print('*'*10 + 'Grad-CAM of Mis Classified Images with respect to Actual(correct) Class' + '*'*10)
        plot_act_cam(cam_no_images, display_layer)