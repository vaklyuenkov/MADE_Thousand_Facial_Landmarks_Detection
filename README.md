# Thousand_Facial_Landmarks_Detection

## Lauch code

```
python hack_train.py --name "finetuned_resnext" --data '/content/data' --gpu --epochs=2 --batch-size=64 -lr=0.005 --gamma=0.2 --checkpoint='v1_resnext_6_best.pth'
```

## How was the model trained

1. 3 epoch using `lr=0.001`, `gamma=1` to get 
2. 3 epoch with `lr=0.001`, `gamma=0.2` to get 
3. 1 epoch with freezed feature extractor

<details>
<summary>Freezed feature extractor</summary>
<br>
  
```python
    model = models.resnext101_32x8d(pretrained=True)
    
#   Uncomment for learning with freezed feature extractor
#   for param in model.parameters():
#     param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True) 
```
</details>


## Submissions
![best submission](pic/submits.png)


## Some attempts

```python
#    model = models.wide_resnet101_2(pretrained=True)
#     fc_layers = nn.Sequential(
#                 nn.Linear(model.fc.in_features, model.fc.in_features),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(p=0.1),
#                 nn.Linear(model.fc.in_features,  2 * NUM_PTS),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(p=0.1))
#     model.fc = fc_layers
```
