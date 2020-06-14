import torch

def convertModel(model,modelName,img_sz):
    print("Start...")
    device = torch.device('cuda')
    #model = torch.load(SavePathModel,map_location=device)
    #model = model.float()
    #model.eval()
    example = torch.rand(1, 3, img_sz[0],img_sz[1],device=device)
    model.eval()
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(modelName)
    print("Complete")