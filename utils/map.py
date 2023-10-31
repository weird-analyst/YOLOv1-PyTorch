import torch
from collections import Counter
from iou import IoU

def mean_average_precision(
        pred_boxes, #All the prediction boxes over the entire batch of images
        true_boxes,
        iou_threshold,
        box_format = "corners",
        num_classes=20):
    
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    #pred_boxes = [[train_idx, class_pred, prob_score, x1, y1, x2, y2],...]
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        #For each class take in all the predictions and truth of that class
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        #Counter will create a dictionary of number of bounding boxes in each train_idx
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            #We will replace the number of bounding boxes with a tensor of zeros of the same shape
            #This is to track which bboxes in the target we have covered in the predictions
            amount_bboxes[key] = torch.zeros(val)

        #Sort the detections wrt confidence
        detections.sort(key=lambda x: x[2], reverse=True)

        #To keep track of if the element is TP or FP
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))

        total_true_bboxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0] 
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = IoU(torch.tensor(detection[3:]),
                          torch.tensor(gt[3:]),
                          box_format=box_format)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            #Now we have the best matching target of the same class as the detection in 1 target image
            #If that best box is below something then that box is a false positive!

            if best_iou > iou_threshold:
                #We also need to check if we have already seen that bbox
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        #Now to calculate the precision and recall for 1 class
        TP_cumsum = torch.cumsum(TP, dim=0)
        #cumsum returns a list of cumsum till that index
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes+epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))

        #While plotting the graph we want to start from 1
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        #getting area using trapx
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)