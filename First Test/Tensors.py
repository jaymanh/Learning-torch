import torch
import numpy as np


#Initializing a Tensor
def FromData():
    data = [[1, 2],[3, 4]]
    x_data = torch.tensor(data)

def FromNumpy():
    data = np.array([[1, 2],[3, 4]])
    x_np = torch.from_numpy(data)

def FromAnotherTensor(x_data):
    x_ones = torch.ones_like(x_data) # retains the properties of x_data
    print(f"Ones Tensor: \n {x_ones} \n")
    
    x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
    print(f"Random Tensor: \n {x_rand} \n")

def WithRandomOrConstantValues():
    shape = (2,3,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")

def AttributesofTensor():
    tensor = torch.rand(3,4)
    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")
    if torch.cuda.is_available():
        tensor = tensor.to('cuda')
        print(f"Device tensor is now stored on: {tensor.device}")


    #Standard numpy-like indexing and slicing
    tensor2 = torch.ones(4, 4)
    print(f"First row: {tensor2[0]}")
    print(f"First column: {tensor2[:, 0]}")
    print(f"Last column: {tensor2[..., -1]}")
    tensor2[:,1] = 0
    if torch.cuda.is_available():
        tensor2 = tensor2.to('cuda')
        print(f"Device tensor is now stored on: {tensor2.device}")
    print(tensor2)

    #Joining tensors
    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    print(t1)

    return t1

def ArithmeticOperations(tensor):
    if torch.cuda.is_available():
        tensor = tensor.to('cuda')
        print(f"Device tensor is now stored on: {tensor.device}")
    #This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
    #''tensor.T'' returns the transpose of tensor.
    y1 = tensor @ tensor.T
    y2 = tensor.matmul(tensor.T)

    y3 = torch.rand_like(tensor)
    torch.matmul(tensor, tensor.T, out=y3)

    #This computes the element-wise product. z1, z2, z3 will have the same value
    z1 = tensor * tensor
    z2 = tensor.mul(tensor)

    z3 = torch.rand_like(tensor)
    torch.mul(tensor, tensor, out=z3)

    print(y1, y2, y3)
    print(z1, z2, z3)

def SingleElementTensor(tensor):
    aggr = tensor.sum()
    print(aggr)
    aggr_item = aggr.item()
    print(aggr_item, type(aggr_item))

    return aggr

def InPlaceOperations(tensor):
    #In-place operations Operations that store the result into the operand are called in-place. They are denoted by a _ suffix. For example: x.copy_(y), x.t_(), will change x.
    # adds tensor to tensor
    print(tensor, "\n")
    tensor.add_(5)
    print(tensor)
    #In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss of history. Hence, their use is discouraged.

#Bridge with NumPy
#Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.
def TensorToNumpy():
    t = torch.ones(5)
    print(f"t: {t}")
    n = t.numpy()
    print(f"n: {n}")

    t.add_(1)
    print(f"t: {t}")
    print(f"n: {n}")

def NumpyToTensor():
    n = np.ones(5)
    t = torch.from_numpy(n)

    np.add(n, 1, out=n)
    print(f"t: {t}")
    print(f"n: {n}")





if __name__ == "__main__":
    FromData()
    FromNumpy()
    FromAnotherTensor(torch.tensor([[5, 6],[7, 8]]))
    WithRandomOrConstantValues()
    AttributesofTensor()
    ArithmeticOperations(torch.rand(3,3))
    SingleElementTensor(torch.rand(3,3))
    InPlaceOperations(torch.rand(3,3))
    TensorToNumpy()