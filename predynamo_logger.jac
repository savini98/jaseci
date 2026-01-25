import torch;

@torch.compile()
def print_func(a:int){
    print(a);
    a = a+1;
    return a;
}