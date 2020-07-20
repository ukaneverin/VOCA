"""
Sample code for fixing partial parameters -------------
"""
pretrained_dict = torch.load(model_directory+'trained_cell_vgg11_cls_%s.pth' % overlap_percentage)
model_dict = model.state_dict()
# overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. load the new state dict
model_ft.load_state_dict(model_dict)
for param in model_ft.features.parameters():
	param.requires_grad = False
for param in model_ft.classifier.parameters():
	param.requires_grad = False
if use_gpu:
	model_ft = model_ft.cuda()
optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.0001, momentum=0.9)