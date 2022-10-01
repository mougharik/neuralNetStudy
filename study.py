import argparse
import tensorflow as tf
import datasetops as do
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import *
from utils import *

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.max(1)[1]
        loss = F.nll_loss(output, target)
        losses.append(loss)
        loss.backward()
        optimizer.step()
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    
    acc = 100. * correct / total
    avg_loss = sum(losses) / len(train_loader)

    print(acc)
    print(avg_loss)
    return acc, avg_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return (100. * correct / len(test_loader.dataset)), test_loss


def main():
    parser = argparse.ArgumentParser(description='NNs for USPS Handwritten Dataset')
    parser.add_argument('--net', type=int, default=1, metavar='N',
                        help='choice of NN to use (default: 1): \
                            \n\t1 - Fully Connected Network \
                            \n\t2 - Locally Connected Network \
                            \n\t3 - Convolutional Network')
    parser.add_argument('--init', type=int, default=1, metavar='N',
                        help='choice of weight initialization (default: 1): \
                            \n\t1 - Effective Learning \
                            \n\t2 - Too fast \
                            \n\t3 - Too slow \
                            \n\t4 - Default of model')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=5e-2, metavar='LR',
                        help='learning rate (default: 5e-2)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='D',
                        help='dropout (default: 0)')
    parser.add_argument('--gamma', type=float, default=0.85, metavar='G',
                        help='Learning rate step gamma (default: 0.85)')
    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                        help='momentum (default: 0.0)')
    parser.add_argument('--viz-filters', action='store_true', default=False,
                        help='visualize maps each layer in MNIST CNN')
    parser.add_argument('--viz-acts', action='store_true', default=False,
                        help='visualize 3rd layer activations for 0 and 8 in MNIST CNN')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    data_train, train_pixels = buildUSPSDataset('data/zip_train.txt', args.net)
    data_test, test_pixels = buildUSPSDataset('data/zip_test.txt', args.net)

    modelChoice = {'1': FCNet, '3': ConvNet}
    
    if args.net != 2:
        model = modelChoice[str(args.net)](args).to(device)
        optim = torch.optim.SGD(model.parameters(), lr=args.lr)

        count_parameters(model)

        train_loader = DataLoader(dataset=data_train, shuffle=True, **train_kwargs)
        test_loader = DataLoader(dataset=data_test, shuffle=False, **test_kwargs)

        writer = SummaryWriter('runs/' 
                + model._get_name() 
                + '/train' 
                + '/init' + str(args.init) 
                + '/lr' + format(args.lr, '.1e').replace('0','').replace('.',''))
        writer2 = SummaryWriter('runs/' 
                + model._get_name() 
                + '/test'
                + '/init' + str(args.init) 
                + '/lr' + format(args.lr, '.1e').replace('0','').replace('.',''))

        scheduler = StepLR(optim, step_size=1)
        for epoch in range(1, args.epochs + 1):
            train_acc, train_loss = train(args, model, device, train_loader, optim, epoch)
            test_acc, test_loss = test(model, device, test_loader)
            scheduler.step()

            writer.add_scalar('loss', train_loss, epoch)
            writer2.add_scalar('loss', test_loss, epoch)

            writer.add_scalar('accuracy', train_acc, epoch)
            writer2.add_scalar('accuracy', test_acc, epoch)


        if args.save_model:
            torch.save(model.state_dict(), 
                model._get_name()
                +'_init' + str(args.init)
                + '_lr' + format(args.lr, '.1e').replace('0','').replace('.',''))

    else:
        model = LCNet(args)
        model.summary()
        
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=args.lr), 
                    loss=tf.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])

        data_train = do.from_pytorch(data_train).to_tensorflow().batch(args.batch_size)
        data_test = do.from_pytorch(data_test).to_tensorflow().batch(args.test_batch_size)
        model.fit(data_train, epochs=args.epochs, validation_data=data_test)
        model.evaluate(data_test)

if __name__ == '__main__':
    main()
