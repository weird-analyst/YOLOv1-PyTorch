import torch
from iou import IoU

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