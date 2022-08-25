vector_thermal_gram[batch_idx] = 

Variable(
    
    
    thermal_gram[batch_idx][torch.triu(torch.ones(thermal_gram[batch_idx].size()[0], thermal_gram[batch_idx].size()[1])) == 1]
    
    thermal_gram[batch_idx] ---> 64 x 64
    
    [torch.triu(torch.ones(64, 64))] == 1


    
    , requires_grad=True)