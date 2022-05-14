import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

import torch
from torch import nn, optim
from torchvision import transforms

import cnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_path = 'data/'

data_dirs = {
    'train': dataset_path+'train/',
    'valid': dataset_path+'valid/',
    'test': dataset_path+'test/',
}

files = os.listdir(data_dirs['train'])
files = files[:30]
idx_to_class = sorted([f.replace('_', ' ').split('.')[0] for f in files])

class_to_idx = {idx_to_class[i]: i for i in range(len(idx_to_class))}


# data transform block
data_transform = {
    'train' : transforms.Compose([
                                  transforms.RandomRotation(10),
                                  transforms.ToTensor()
    ]),

    'valid' : transforms.Compose([
                                  transforms.ToTensor()
    ]),
 
    'test' : transforms.Compose([
                                 transforms.ToTensor()
    ])

}

# dataset block
dataset = {}

for d in ['train', 'valid', 'test']:
    data_x = []
    data_y = []

    for path, _, temp in os.walk(data_dirs[d]):
      for f in files:
        c = f.replace('_', ' ').split('.')[0]
        x = np.load(path + f).reshape(-1, 28, 28) / 255
        y = np.ones((len(x), 1), dtype=np.int64) * class_to_idx[c]

        data_x.extend(x)
        data_y.extend(y)

      dataset[d] = torch.utils.data.TensorDataset(torch.stack([data_transform[d](Image.fromarray(np.uint8(i*255))) for i in data_x]), 
                                                torch.stack([torch.Tensor(j) for j in data_y]))
      

#batch size
bs = 128

dataloaders = {
    d: torch.utils.data.DataLoader(dataset[d], batch_size=bs, shuffle=True) for d in ['train', 
                                                                                      'valid', 
                                                                                      'test']
}


for data_idx, (data,target) in enumerate(dataloaders['test']):
  print(data.shape)
  if data_idx==4:
    break


dataiter = iter(dataloaders['train'])
images, labels = dataiter.next()

fig = plt.figure(figsize=(15, 15))
for cls in np.arange(30):
    ax = fig.add_subplot(5, 6, cls+1, xticks=[], yticks=[])
    image = images.numpy()[labels.numpy().reshape(-1) == cls][0]
    label = labels.numpy()[labels.numpy().reshape(-1) == cls][0, 0].astype(np.int64)
    plt.imshow(image.reshape(28,28), cmap='gray')
    ax.set_title(idx_to_class[label])
plt.savefig('data_samples.png')


model_scratch = cnn.Net()
model_scratch.to(device)


criterion_scratch = nn.CrossEntropyLoss()
optimizer_scratch = optim.Adam(model_scratch.parameters(), lr = 0.001)

# train and validate cycle
def train(epochs, loaders, model, optimizer, criterion, device, save_path):
    #pointer for min validation loss
    valid_loss_min = np.Inf 
    train_losses = []
    val_losses = []

    for epoch in range(1,epochs+1):
        #pointer 
        train_loss =  0.0
        valid_loss = 0.0

        #model in training mode
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            #move to GPU
            data, target = data.to(device) , target.long().to(device)

            #5 steps
            optimizer.zero_grad() # make sure's grads don't pile up
            output = model(data) # predictions
            loss = criterion(output, torch.max(target,1)[0]) #loss
            loss.backward() #backward propagation
            optimizer.step() #updation!

            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            #model in validation mode
            model.eval()
        for batch_idx, (data,target) in enumerate(loaders['valid']):
            # move to GPU
            data, target = data.to(device), target.long().to(device)
            
            #update average val loss
            output = model(data)
            loss = criterion(output,torch.max(target,1)[0])
            valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        #print stats
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(    
            epoch, 
            train_loss,
            valid_loss
            ))
        
        train_losses.append(train_loss)
        val_losses.append(valid_loss)

        if valid_loss < valid_loss_min:
          print('Saving model...')
          valid_loss_min = valid_loss
          torch.save(model.state_dict(), save_path)

    #finally return model
    return model, train_losses, val_losses



# train the model
model_scratch, train_losses, val_losses = train(25, dataloaders, model_scratch, optimizer_scratch, 
                                                criterion_scratch, device, 'model_25.pt')



plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xticks([i for i in range(0, len(train_losses), 5)].append(len(train_losses)))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('tr_val.png')
_ = plt.ylim()
plt.show()

#load the model
model_scratch.load_state_dict(torch.load('model_25.pt'))

def mapk(target, output, k=3):
    map_sum = 0
    output = torch.topk(output, k)[1]
    for i, t in enumerate(target):
        idx = (output[i] == t).nonzero().cpu().numpy()
        if len(idx) == 0:
            continue
            
        idx = idx[0][0]
        map_sum += 1 / (idx + 1)
        
    return map_sum


def test(loaders, model, criterion, device):    
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.
    y = None
    y_pred = None
    total_mapk = 0
    
    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU or CPU
        data, target = data.to(device), target.long().to(device)
        
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target.view(-1))
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        
        if y is None:
            y = target.cpu().numpy()
            y_pred = pred.data.cpu().view_as(target).numpy()
            probs = output.data.cpu().numpy()
        else:
            y = np.append(y, target.cpu().numpy())
            y_pred = np.append(y_pred, pred.data.cpu().view_as(target).numpy())
            probs = np.vstack([probs, output.data.cpu().numpy()])
            
        total_mapk += mapk(target, output)
        
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
    
    map3 = np.around(total_mapk / len(loaders["test"].dataset) * 100, 2)
    print(f'\nMean Average Precision @ 3: {map3}%')
          
    return y, y_pred, probs

y, y_pred, probs = test(dataloaders, model_scratch, criterion_scratch, device)
