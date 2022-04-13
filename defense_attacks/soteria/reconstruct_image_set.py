"""Run reconstruction in a terminal prompt.

Optional arguments can be found in inversefed/options.py
"""

import torch
import torchvision

import numpy as np
from PIL import Image

import inversefed
torch.backends.cudnn.benchmark = inversefed.consts.BENCHMARK

from collections import defaultdict
import time
import os

# Parse input arguments
args = inversefed.options().parse_args()
# Parse training strategy
defs = inversefed.training_strategy('conservative')
defs.epochs = args.epochs


if __name__ == "__main__":
    # Choose GPU device and print status information:
    setup = inversefed.utils.system_startup(args)
    start_time = time.time()

    use_steps = args.use_steps
    # Prepare for training

    # Get data:
    loss_fn, trainloader, validloader = inversefed.construct_dataloaders(args.dataset, defs, data_path=args.data_path)

    dm = torch.as_tensor(getattr(inversefed.consts, f'{args.dataset.lower()}_mean'), **setup)[:, None, None]
    ds = torch.as_tensor(getattr(inversefed.consts, f'{args.dataset.lower()}_std'), **setup)[:, None, None]

    if args.dataset == 'ImageNet':
        if args.model == 'ResNet152':
            model = torchvision.models.resnet152(pretrained=args.trained_model)
        else:
            model = torchvision.models.resnet18(pretrained=args.trained_model)
        model_seed = None
    else:
        model, model_seed = inversefed.construct_model(args.model, num_classes=10, num_channels=3)

    # Set paths


    # Load state dict
    trained = True
    if trained:
        if use_steps:
            path = f"models/{args.model}_bs_32_steps_{args.steps}.pth"
            out_folder = f'results/steps_{args.steps}_{args.model}-{args.defense}-{args.pruning_rate}-tv-{args.tv}'
        else:
            path = f"models/{args.model}_{args.steps}.pth"
            out_folder = f'results/epochs_{args.steps}_{args.model}-{args.defense}-{args.pruning_rate}-tv-{args.tv}'

        s_dict = torch.load(path)

        model.load_state_dict(s_dict)
    else:
        out_folder = f'results/untrained_{args.model}-{args.defense}-{args.pruning_rate}-tv-{args.tv}'

    model.to(**setup)
    model.eval()

    # Sanity check: Validate model accuracy
    training_stats = defaultdict(list)
    # inversefed.training.training_routine.validate(model, loss_fn, validloader, defs, setup, training_stats)
    # name, format = loss_fn.metric()
    # print(f'Val loss is {training_stats["valid_losses"][-1]:6.4f}, Val {name}: {training_stats["valid_" + name][-1]:{format}}.')

    os.makedirs(out_folder, exist_ok=True)

    num_examples = 100

    rec_stats = list()

    for idx in range(num_examples):
        print(f"Running example {idx}")
        ground_truth, labels = trainloader.dataset[idx]
        ground_truth, labels = ground_truth.unsqueeze(0).to(**setup), torch.as_tensor((labels,), device=setup['device'])

        img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])


        model.zero_grad()
        ground_truth.requires_grad = True
        target_loss, _, _ = loss_fn(model(ground_truth), labels)

        if args.model in ['LeNet', 'ConvNet', 'ConvBig'] and args.defense == 'ours':
            feature_fc1_graph = model.extract_feature()
            deviation_f1_target = torch.zeros_like(feature_fc1_graph)
            deviation_f1_x_norm = torch.zeros_like(feature_fc1_graph)
            for f in range(deviation_f1_x_norm.size(1)):
                deviation_f1_target[:,f] = 1
                feature_fc1_graph.backward(deviation_f1_target, retain_graph=True)

                deviation_f1_x = ground_truth.grad.data
                
                deviation_f1_x_norm[:,f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1)/(feature_fc1_graph.data[:,f] + 0.1)
                
                model.zero_grad()

                ground_truth.grad.data.zero_()
                
                deviation_f1_target[:,f] = 0
            deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
            thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().numpy(), args.pruning_rate)
            mask = np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)


        input_gradient = torch.autograd.grad(target_loss, model.parameters())
        input_gradient = [grad.detach() for grad in input_gradient]
        if args.model == 'LeNet' and args.defense == 'ours':
            input_gradient[8] = input_gradient[8] * torch.Tensor(mask).to(**setup)
        elif args.model == 'ConvNet' and args.defense == 'ours':
            input_gradient[-2] = input_gradient[-2] * torch.Tensor(mask).to(**setup) 
        elif args.model == 'ConvBig' and args.defense == 'ours':
            # NOTE: This is our defense/attack on the ConvBig network. If you want to defend another layer please follow these steps:
            # 1. Select your model in inversefed/nn/models.py
            # 2. Adapt your selected model in inversefed/nn/models.py by letting the layer before the defended layer ouput to self.feature (See our ConvBig)
            # 3. Change the index below to represent the layer'S weights. Note that -1 = last layer bias, -2 = last layer weights, -3 = second to last layer bias ...
            # 4. If you want to use our attack on layer k do the following
            #       - Go to inversefed/reconstruction_algorithms.py Line 372 (Marked)
            #       - Set remove_defense to True
            #       - Set attack_idx to k 
            input_gradient[-4] = input_gradient[-4] * torch.Tensor(mask).to(**setup) 
        if args.defense == 'prune':
            for i in range(len(input_gradient)):
                grad_tensor = input_gradient[i].cpu().numpy()
                flattened_weights = np.abs(grad_tensor.flatten())
                # Generate the pruning threshold according to 'prune by percentage'. (Your code: 1 Line) 
                thresh = np.percentile(flattened_weights, args.pruning_rate)
                grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
                input_gradient[i] = torch.Tensor(grad_tensor).to(**setup)
        full_norm = torch.stack([g.norm() for g in input_gradient]).mean()
        #print(f'Full gradient norm is {full_norm:e}.')

        config = dict(signed=args.signed,
                    boxed=args.boxed,
                    cost_fn=args.cost_fn,
                    indices='def',
                    weights='equal',
                    lr=0.1,
                    optim=args.optimizer,
                    restarts=args.restarts,
                    max_iterations=4_000,
                    total_variation=args.tv,
                    init='randn',
                    filter='none',
                    lr_decay=True,
                    scoring_choice='loss')
        rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=args.num_images)
        output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=img_shape, dryrun=args.dryrun)

        # Compute stats
        output_denormalized = torch.clamp(output * ds + dm, 0, 1)
        gt_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
        denorm_mse = (output_denormalized - gt_denormalized).pow(2).mean().item()
        denorm_mse_sum = (output_denormalized - gt_denormalized).pow(2).sum().item()
        test_mse = (output - ground_truth).pow(2).mean().item()
        feat_mse = (model(output) - model(ground_truth)).pow(2).mean().item()
        test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1 / ds)

        # Save the resulting image       
        rec_filename = f'rec_{idx}.png'
        torchvision.utils.save_image(output_denormalized, os.path.join(out_folder, rec_filename))

        gt_filename = (f'ori_{idx}.png')
        torchvision.utils.save_image(gt_denormalized, os.path.join(out_folder, gt_filename))

        # Save to a table:
        print(f"Image: {idx} Def_stat: {args.pruning_rate} TV: {args.tv} Rec. loss: {stats['opt']:2.4f} | Delta: {denorm_mse_sum:.4f} | 0-1-MSE: {denorm_mse:.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |")
        rec_stats.append({
            'delta': denorm_mse_sum,
            'mse': test_mse,
            'denorm_mse': denorm_mse,
            'psnr': test_psnr,
            'fmse': feat_mse
        })
    
    np.save(f'{out_folder}/metric.npy', rec_stats)