import torch


class Metric:
    def __init__(self, metric='dice'):
        self.metric = metric

    def __call__(self, output, label):
        if self.metric == 'dice':
            return dice_func(output, label)
        elif self.metric == 'top1':
            return top1_acc(output, label)


def dice_func(output, label):
    softmax_pred = torch.nn.functional.softmax(output, dim=1)
    seg_pred = torch.argmax(softmax_pred, dim=1)
    all_dice = 0
    label = label.squeeze(dim=1)
    batch_size = label.shape[0]
    num_class = softmax_pred.shape[1]
    for i in range(num_class):
        each_pred = torch.zeros_like(seg_pred)
        each_pred[seg_pred == i] = 1

        each_gt = torch.zeros_like(label)
        each_gt[label == i] = 1

        intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)

        union = each_pred.view(batch_size, -1).sum(1) + each_gt.view(batch_size, -1).sum(1)
        dice = (2. * intersection) / (union + 1e-5)

        all_dice += torch.mean(dice)

    return all_dice.item() * 1.0 / num_class


def top1_acc(output, label):
    total = label.size(0)
    pred = output.data.max(1)[1]
    correct = pred.eq(label.view(-1)).sum().item()

    return correct * 1.0 / total