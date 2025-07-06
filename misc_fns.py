import torch

def get_dataset(df, window_size=3, features = [ "Open", "High", "Low", "Close", "Adj Close" ]):
    df = df.sort_values('Date')
    data = torch.tensor(df[features].values, dtype=torch.float32)
    inputs, outputs = [], []

    for i in range(window_size, len(data)):
        input_window = data[i - window_size:i].reshape(-1)     
        output_point = data[i]                                  
        
        inputs.append(input_window)
        outputs.append(output_point)
    
    inputs_tensor  = torch.stack(inputs)
    outputs_tensor = torch.stack(outputs)

    return(inputs_tensor, outputs_tensor)


def forward(inputs, model):
    """
    inputs: (seq_len, input_dim)
    model: IOHMM_NN

    outputs: (seq_len, input_dim)
    """
    seq_len, _ = inputs.size()
    state_prob = None
    outputs = []
    for t in range(seq_len):
        input_t = inputs[t,:]
        output_t, state_prob = model(input_t, state_prob)
        outputs.append(output_t)
    
    outputs = torch.stack(outputs)
    return outputs

def optimize(model, inputs, targets, optimizer, loss_fn, num_epochs=1000, print_loss=True, print_freq=100):
    """
    inputs: (seq_len, input_dim)
    targets: (seq_len, output_dim)
    model: IOHMM_NN
    optimizer: torch.optim.Optimizer
    loss_fn: callable
    num_epochs: int
    print_loss: bool
    """
    for epoch in range(1, 1 + num_epochs):
        optimizer.zero_grad()
        outputs = forward(inputs, model)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        if print_loss and epoch % print_freq == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            
def performance(model, inputs, targets, loss_fn):
    outputs = forward(inputs, model)
    
    loss     = loss_fn(targets, outputs)
    r_square = 1 - torch.sum((targets-outputs)**2)/torch.sum((targets - torch.mean(targets)) ** 2)
    
    return { "loss": loss.item(), "r_square": r_square.item() }