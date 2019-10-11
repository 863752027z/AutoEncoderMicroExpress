import torch
x = torch.rand(5, 5, requires_grad=True)
y = torch.rand(5, 5, requires_grad=True)
print(x)

z = x**2 + y**2
print(z.grad_fn)
z.backward(torch.ones_like(x))
print(x.grad)
print(y.grad)
print(z.is_leaf)
print(x.is_leaf)
print('z.grad_fn:' + str(z.grad_fn))
print('z.grad_fn.next_functions:' + str(z.grad_fn.next_functions))
xg = z.grad_fn.next_functions[0][0]

#扩展Autograd
#有些运算不能自动求导，需要我们自己定义
from torch.autograd.function import Function
class MulConstant(Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        ctx.constant = constant
        return tensor*constant
    @staticmethod
    def backward(ctx, grad_output):
        #返回的参数要与输入的参数一样
        #第一个输入是3x3的张量，第二个参数是常量
        #常数的梯度必须是None
        return grad_output, None

a = torch.rand(3, 3, requires_grad=True)
b = MulConstant.apply(a, 5)
print(a)
print(b)
b.backward(torch.zeros_like(a))
print(a.grad)