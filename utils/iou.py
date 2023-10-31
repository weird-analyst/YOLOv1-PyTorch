import torch

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