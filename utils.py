import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def IoU(boxes_preds, boxes_labels, box_format = "midpoint"):
    #The boxes_preds may have multiple boxes for a batch of images/predictions
    #The shape of boxes_preds and boxes_labels is (N,4)    
    if box_format == "midpoint": #Assuming (x, y, w, h)
        box1_x1 = boxes_preds[..., 0:1] - (boxes_preds[..., 2:3]/2)
        box1_y1 = boxes_preds[..., 1:2] - (boxes_preds[..., 3:4]/2)
        box1_x2 = boxes_preds[..., 0:1] + (boxes_preds[..., 2:3]/2)
        box1_y2 = boxes_preds[..., 1:2] + (boxes_preds[..., 3:4]/2)
        box2_x1 = boxes_labels[..., 0:1] - (boxes_labels[..., 2:3]/2)
        box2_y1 = boxes_labels[..., 1:2] - (boxes_labels[..., 3:4]/2)
        box2_x2 = boxes_labels[..., 0:1] + (boxes_labels[..., 2:3]/2)
        box2_y2 = boxes_labels[..., 1:2] + (boxes_labels[..., 3:4]/2)
    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1] #We are using splicing to retain dimensions
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    else:
        raise ValueError("box_format can only be midpoint or corners")
    #Calculating coordinates of Intersection box
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    #Calculating area of intersection
    #.clamp(0) is for handling non intersection case
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    #Calculating union and intersection over union
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
 
    union = box1_area + box2_area - intersection
    
    return intersection/union


def nms(
    bboxes, #input bounding boxes
    iou_threshold,
    threshold,
    box_format = "corners"
):
    #The final output predictions will be a list of bounding boxes, each element of this
    #list will have 6 values, the class, it's probability and the bounding box coords
    
    assert type(bboxes) == list
    
    #Take all the bboxes which have confidence greater than some threshold
    bboxes = [box for box in bboxes if box[1]>threshold]

    bboxes_after_nms = []
    
    #Sorting and taking the largest confidence
    bboxes = sorted(bboxes, key= lambda x: x[1], reverse=True)
    
    while bboxes:
        #Pop the box with the highest confidence
        chosen_box = bboxes.pop(0)
        
        #Now iterate through rest of the boxes and check the IoU with boxes of same class'
        #What you do is: go through the remaining boxes, take the ones of the different class,
        #then take the ones which point to different objects of same class (<iou_threshold)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or IoU(torch.tensor(chosen_box[2:]),
                   torch.tensor(box[2:]),
                   box_format = box_format) < iou_threshold 
        ]
        
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


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




def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = nms(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )


            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes



def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])