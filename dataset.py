from abc import ABC, abstractmethod, abstractproperty
import torch
from torch.utils.data import Dataset, DataLoader

class AbsBinaryData(ABC, Dataset):
    def __init__(self, dim, num) -> None:
        """Represents a data object that holds binary data.

        Args:
            dim (_type_): dimensionality of the x
            num (_type_): number of samples
        """
        super().__init__()
        self.dim = dim
        self.num = num
        
        self.x1 = self.gen_x()
        self.x2 = self.gen_x()

        self.y = None
        
    def __len__(self):
        return self.num
    
    def gen_x(self):
        """Generates the x."""
        # generate binary array of shape (num, dim)
        return torch.randint(2, size=(self.num, self.dim))
    
    def x_to_str(self, x):
        """Converts the x to a string."""
        return ''.join([str(i) for i in x])
    
    @abstractmethod
    def gen_y(self):
        """Generates the y for the x."""
        raise NotImplementedError
    
    def binary_to_decimal(self, binary_tensor):
        """Converts a binary tensor to a decimal tensor.

        Args:
            binary_tensor (_type_): _description_

        Returns:
            _type_: _description_
        """
        decimal_tensor = torch.zeros(binary_tensor.size(0), dtype=torch.int)
        for i in range(binary_tensor.size(1)):
            decimal_tensor = decimal_tensor * 2 + binary_tensor[:, i]
        return decimal_tensor
    
    def decimal_to_binary(self, decimal_tensor):
        """converts a decimal tensor to a binary tensor.

        Args:
            decimal_tensor (_type_): _description_

        Returns:
            _type_: _description_
        """
        binary_tensor = torch.zeros(decimal_tensor.size(0), self.dim, dtype=torch.int)
        for i in range(self.dim - 1, -1, -1):
            binary_tensor[:, i] = decimal_tensor % 2
            decimal_tensor = decimal_tensor // 2        
        assert binary_tensor.size() == (len(decimal_tensor), self.dim), "size mismatch"
        return binary_tensor

    
    def __getitem__(self, index):
        raise NotImplementedError

class BinaryAddData(AbsBinaryData):
    def __init__(self, dim, num, msb_first=True) -> None:
        super().__init__(dim, num)
        self.y = self.gen_y()
        self.msb_first = msb_first # determines the first bit that is input to the rnn
    
    def gen_y(self):
        y = self.binary_to_decimal(self.x1) + self.binary_to_decimal(self.x2)
        return self.decimal_to_binary(y)

    def __getitem__(self, index):
        x1 = self.x1[index]
        x2 = self.x2[index]
        x = torch.stack([x1.view(1, -1), x2.view(1, -1)], dim=0).T.squeeze()
        if self.msb_first == False:
            x = torch.flip(x, dims=[1])
        y = self.y[index]
        return x, y