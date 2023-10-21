import numpy as np
import os
import pickle


def save_model_ckpt(model, ckpt_dir, model_name) -> None:
    model_ckpt = {
        "params": {},
        "num_epochs": 0,
        "loss": 0
    }
    model_ckpt["params"]["U"] = model.U
    model_ckpt["params"]["V"] = model.V
    model_ckpt["params"]["W"] = model.W
    model_ckpt["params"]["b_hidden"] = model.b_hidden
    model_ckpt["params"]["b_out"] = model.b_out
    
    with open(f"{os.path.join(ckpt_dir, model_name)}", "wb") as f_out:
        pickle.dump(model_ckpt, f_out)


def load_model_ckpt(model, ckpt_dir, model_name):
    if not os.path.exists(ckpt_dir):
        raise IOError("Model checkpoint not found.")
    else:
        with open(f"{os.path.join(ckpt_dir, model_name)}", "rb") as f_in:
            model_ckpt = pickle.load(f_in)
        
        model.U = model_ckpt["params"]["U"]
        model.V = model_ckpt["params"]["V"]
        model.W = model_ckpt["params"]["W"]
        model.b_hidden = model_ckpt["params"]["b_hidden"]
        model.b_out = model_ckpt["params"]["b_out"]
        
        return model, model_ckpt


def init_orthogonal(param):
    """
    Initializes weight parameters orthogonally.
    Refer to this paper for an explanation of this initialization:
    https://arxiv.org/abs/1312.6120
    """
    if param.ndim < 2:
        raise ValueError("Only parameters with 2 or more dimensions are supported.")

    rows, cols = param.shape
    new_param = np.random.randn(rows, cols)
    if rows < cols:
        new_param = new_param.T
    
    # Compute QR factorization
    q, r = np.linalg.qr(new_param)
    
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = np.diag(r, 0)
    ph = np.sign(d)
    q *= ph

    if rows < cols:
        q = q.T
    
    new_param = q
    
    return new_param


def sigmoid(x, derivative=False):
    x_safe = x + 1e-12
    f = 1 / (1 + np.exp(-x_safe))
    if derivative: # Return the derivative of the function evaluated at x
        return f * (1 - f)
    else: # Return the forward pass of the function at x
        return f
    
    
def tanh(x, derivative=False):
    x_safe = x + 1e-12
    f = (np.exp(x_safe)-np.exp(-x_safe))/(np.exp(x_safe)+np.exp(-x_safe))
    if derivative: # Return the derivative of the function evaluated at x
        return 1-f**2
    else: # Return the forward pass of the function at x
        return f
    

def softmax(x, derivative=False):
    x_safe = x + 1e-12
    f = np.exp(x_safe) / np.sum(np.exp(x_safe))
    if derivative: # Return the derivative of the function evaluated at x
        pass # We will not need this one
    else: # Return the forward pass of the function at x
        return f