
import torch 
from torch import nn, Tensor

from tqdm import tqdm

from plotter import LivePlotter
from torch_common import cpu, gpu
from torch_common import *


import torchattacks 



atanh = lambda x: .5* torch.log((1.+x)/(1.-x))
def _get_CW_inv_optim_space(x:Tensor)->Tensor:
    return atanh(torch.clamp(x*2.-1,-.999,.999)).to(x.device)#.requires_grad_()

def _get_CW_optim_space(x:Tensor):
    #return 1./ (2.* (torch.tanh(x)+1.))
    return .5 * (torch.tanh(x)+1.)

def _CW_lossf(out:Tensor, labels:Tensor, kappa=1, n=10):
    
    y = torch.nn.functional.one_hot(labels, n)
    target_logit = torch.max((1.-y)*out, dim=1)[0]
    correct_logit= torch.max(y*out, dim=1)[0]

    # maximize target, minimize correct
    # 'forget' values less than kappa 
    return torch.clamp((correct_logit-target_logit), min=-kappa).sum()



def evaluate_models_CW_l2(models : list[nn.Module], 
                    xs : Tensor, ys : Tensor, 
                    bsize = 64,
                    n = 50, kappa=1, C = 1,
                    eot_n=5,# eot_bsize=64,
                    lr=5e-3, eps=1,
                    class_n=10,
                    #model_graph_names=["base", "voting"],
                    model_graph_names=None,
                    device=gpu):
    pass;

    assert model_graph_names is None or len(models) == len(model_graph_names)

    xs, ys = xs.to(device), ys.to(device).clone().detach()
    xs = [xs.clone().detach() for _ in models]
    #ys = [ys.clone().detach() for _ in models]
    models = [m.to(device) for m in models]
    shape = xs[0].shape

    if model_graph_names is None:
        model_graph_names = [f"model {i}" for i in range(len(models))]

    plotter = LivePlotter(model_graph_names, [0, 1], errorBarOpacity=0)
    #plotter.loadData("PLOTDATA - Copy.pkl"); quit()
    plotter.xlabel("CW iterations")
    plotter.ylabel("accuracy")
    #plotter.title("")

    orig_xs = xs[0].clone()
    smooth_out = epoch_every = 1
    test_every =  1
    clean_acc_test_max_size = 64

    with torch.no_grad():
        data = []
        for i, model in en(models):
            clean_acc = get_acc(model(
                xs[0][:clean_acc_test_max_size]), 
                ys[:clean_acc_test_max_size])
            print(f"{model_graph_names[i]} clean acc = {clean_acc}")
            data.append(clean_acc.item())

    plotter.addEstimateData(data, False)
    plotter.concludeEpoch()


    # CW inits
    Ws = [_get_CW_inv_optim_space(x).detach().requires_grad_() for x in xs]
    best_advs = [x.clone().detach().to(device) for x in xs]
    optims = [torch.optim.Adam([w], lr=lr) for w in Ws]
    flat, mse = torch.nn.Flatten(), torch.nn.MSELoss(reduction="none")
    best_divs = [torch.ones((xs[0].shape[0])).to(device) * 1e8 for _ in models]
    flat_l2 = lambda a,b: mse(flat(a), flat(b)).sum(dim=1)#.sum()

    for step_i in (pbar:= tqdm(range(n))):
        pass;
        accs = [0 for _ in models]
        inner_iterator = range(0, shape[0], bsize)
        n_ = len(list(inner_iterator))
        for i, xi in en(inner_iterator):
            
            xi_end = min(shape[0], xi+bsize)
            for j, model in en(models):
                pass;
            
                # split xs into largest value that an fit in memory
                x=xs[j][xi:xi_end, :,:,:]
                y=ys[xi:xi_end]
                orig_x = orig_xs[xi:xi_end, :,:,:]
                optim = optims[j]
                W = Ws[j]
                #best_adv = best_advs[j]
                best_div = best_divs[j][xi:xi_end]

                adv = _get_CW_optim_space(W)[xi:xi_end, :,:,:]
                loss_div_per_datapoint = flat_l2(adv, orig_x)
                loss_div = loss_div_per_datapoint.sum()

                loss_adv = 0
                out_avg = None
                # average over runs (eot)
                for _ in range(eot_n):
                    out=model(adv)
                    loss_adv += _CW_lossf(out, y, kappa, class_n) / eot_n
                    out_avg = (out / eot_n) if out_avg is None else (out_avg + (out / eot_n))
                    del out

                #print(loss_adv.shape, loss_div.shape); quit()

                loss:Tensor = loss_div + C * loss_adv
                optim.zero_grad()
                loss.backward()
                optim.step()


                results = torch.argmax(out_avg, dim=1)
                mask = (results!=y).float()
                mask = mask * (best_div > loss_div_per_datapoint.detach())
                best_div = mask * loss_div_per_datapoint.detach() + (1.-mask) * best_div

                best_divs[j][xi:xi_end] = best_div

                mask = mask.view([-1, 1, 1, 1])
                best_advs[j][xi:xi_end, :,:,:] = \
                    mask * adv.detach() + (1.-mask) * best_advs[j][xi:xi_end, :,:,:]

                #eps_mask = (best_advs[j]-orig_x).norm(p=2) <= eps
                #eps_mask = eps_mask.float()
                #best_advs[j] = eps_mask * best_advs[j] + (1.-eps_mask) * orig_x

                #xs[j] = best_advs[j]


            if step_i % test_every == 0:
                with torch.no_grad():
                    pbar.set_description_str("TESTING")
                    data = []

                    for j, model in en(models):
                        x = best_advs[j][xi:xi_end, :,:,:]
                        y=ys[xi:xi_end]
                        out = model(x)
                        acc = (get_acc(out, y))
                        #print(acc)
                        accs[j] += acc
                        data.append(acc)
                        #if j==1:print(acc)

                    plotter.addEstimateData(
                        [x.item() if isinstance(x, Tensor) else x 
                        for x in data],  
                        #i == 1 or i % test_every == 0   
                        True
                    )
                
                    avgL2=0
                    for diff in [(x_-orig_xs) for x_ in best_advs]:
                        sm = sum([x.norm(p=2) for x in diff])
                        #print(diff.norm(p=2,dim=1)); quit()
                        sm = torch.nan_to_num(sm, nan=0.0)
                        avgL2 += sm / diff.shape[0] / len(best_advs)
                    #avgL2 = sum([x_.norm(p=2, dim=1).mean() for x_ in diffs]) / len(xs)

                    pbar.set_description_str("saving")

                    pil_out = torch.cat(best_advs+[orig_xs])
                    #print(pil_out.shape)
                    topil(pil_out.cpu().detach()).save("latest_adv.png")

                    #pbar.set_description_str(f'{i}/{n_} done, avg l2 dist = {round(avgL2.item(), 3)}, accuracies: '+
                    pbar.set_description_str(f'avg l2 dist = {round(avgL2.item(), 3)}, accuracies: '+
                        ', '.join([
                            str( round(x.item() / (i+1), 3) ) 
                            for x in accs]))



        if step_i % epoch_every == 0: 
            plotter.concludeEpoch(x_inc=epoch_every)

    



















def evaluate_models_PGD_l2(models : list[nn.Module], 
                    xs : Tensor, ys : Tensor, 
                    bsize=2, n = 50, eot_n=5, eot_bsize=64,
                    lrs=5e-3, eps=1, lossfs=[nn.CrossEntropyLoss()],
                    class_n=10,
                    model_graph_names=["base", "voting"],
                    max_clean_acc_test_size=50,
                    device=gpu):
    pass;

    assert len(models) == len(model_graph_names)

    xs, ys = xs.to(device), ys.to(device)
    xs = [xs.clone() for _ in models]
    models = [m.to(device) for m in models]
    shape = xs[0].shape


    plotter = LivePlotter(model_graph_names, [0, 1], errorBarOpacity=.12)
    #plotter.loadData()
    plotter.xlabel("PGD iterations")
    plotter.ylabel("accuracy")
    #plotter.title("")

    orig_xs = xs[0].clone()
    smooth_out = epoch_every = 2
    test_every =  1

    with torch.no_grad():
        data = []
        k_ = max_clean_acc_test_size
        for i, model in en(models):
            clean_acc = get_acc(model(xs[0][:k_]), ys[:k_])
            print(f"clean acc model {i} = {clean_acc}")
            data.append(clean_acc.item())
    
    #quit()

    plotter.addEstimateData(data, False)
    plotter.concludeEpoch()

    if not islist(lossfs):
        lossfs = [lossfs for _ in models]
    if not islist(lrs):
        lrs = [lrs for _ in models]

    for step_i in (pbar:= tqdm(range(n))):

        accs = [0 for _ in models]
        inner_iterator = range(0, shape[0], bsize)
        n_ = len(list(inner_iterator))
        for i, xi in en(inner_iterator):
            
            xi_end = min(shape[0], xi+bsize)
            for j, model in en(models):

                # split xs into largest value that an fit in memory
                x=xs[j][xi:xi_end, :,:,:]
                y=ys[xi:xi_end]
                orig_x = orig_xs[xi:xi_end, :,:,:]
                lossf = lossfs[j]
                lr = lrs[j]


                # average over runs (EOT)
                x.requires_grad = True
                eot_xs = x.repeat(eot_n, 1, 1, 1).split(eot_bsize)
                eot_ys = y.repeat(eot_n).split(eot_bsize)

                for i_ in range(len(eot_xs)):
                    eot_x, eot_y = eot_xs[i_], eot_ys[i_]
                    out = model(eot_x)
                    eot_y = torch.nn.functional.one_hot(eot_y, class_n) \
                            #if not isinstance(lossf, torch.nn.CrossEntropyLoss) else eot_y
                    loss = lossf(out, eot_y.float())
                    loss.backward() # accumulate 

                grad = x.grad / eot_n
                x.requires_grad=False
            
                # update x and write back into xs
                x += lr * grad.sign()
                x = project_l2(x, orig_x, eps)
                #x = project_linf(x, orig_x, eps)
                xs[j][xi:xi_end, :,:,:] = x



            if i % test_every == 0:
                with torch.no_grad():
                    pbar.set_description_str("TESTING")
                    data = []

                    for j, model in en(models):
                        x = xs[j][xi:xi_end, :,:,:]
                        y=ys[xi:xi_end]
                        out = model(x)
                        acc = (get_acc(out, y))
                        #print(acc)
                        accs[j] += acc
                        data.append(acc)
                        #if j==1:print(acc)

                    plotter.addEstimateData(
                        [x.item() if isinstance(x, Tensor) else x 
                        for x in data],  
                        #i == 1 or i % test_every == 0   
                        True
                    )
                
                    avgL2=0
                    for diff in [(x_-orig_xs) for x_ in xs]:
                        avgL2 += sum([x.norm(p=2) for x in diff]) / diff.shape[0] / len(xs)
                    #avgL2 = sum([x_.norm(p=2, dim=1).mean() for x_ in diffs]) / len(xs)

                    pbar.set_description_str(f'{i}/{n_} done, avg l2 dist = {round(avgL2.item(), 3)}, accuracies: '+
                        ', '.join([
                            str( round(x.item() / (i+1.), 3) ) 
                            for x in accs]))


                    pil_out = torch.cat(xs+[orig_xs])
                    #print(pil_out.shape)
                    pil_out = pil_out.clamp(0,1)
                    topil(pil_out.cpu().detach()).save("latest_adv.png")

        if step_i % epoch_every == 0: 
            plotter.concludeEpoch(x_inc=epoch_every)

    


