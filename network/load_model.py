import  torch

def load_network(args):
    if args.modelname == 'Affine':
        from network.Affine import  Affine
        model = Affine(imgsize=args.patch_size,device=args.device)
    elif args.modelname == 'Deformation':
        from network.Maskflownet import MaskFlownet
        model = MaskFlownet(device=args.device)
    else:
        print('No model!!!')
    if args.cpt != None:
        try:
            model.load_state_dict(torch.load(args.cpt, map_location=args.device)['state_dict'])
        except:
            model.load_state_dict(torch.load(args.cpt, map_location=args.device))
    model.to(args.device)
    return model
