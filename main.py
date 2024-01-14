import torch

torch.manual_seed(0)
import torch.nn as nn
from dataset import BinaryAddData
import constants
import logging

logging.basicConfig(
    filename="binary_ops.log",
    format="%(asctime)s :: %(filename)s:%(funcName)s :: %(message)s",
    filemode="a",
)
logger = logging.getLogger(name="binary_ops")
logger.setLevel(logging.DEBUG)

logger.info("****" * 20)
print("inference with more bits")

device = "cuda:6"

params = {
    constants.BSZ: 16,
    constants.NUM_BITS: 10,
    constants.NUM_TRN: 100,  # 'num_samples
    constants.NUM_TST: 100,
    constants.HIDSIZE: 4,
    constants.LR: 0.001,
    constants.NUM_EPOCHS: 2000,
}


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        # Define an RNN layer
        self.rnn = nn.RNN(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )

        # Define a fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        # Input sequence shape: (batch_size, sequence_length, input_size)
        # Output shape: (batch_size, sequence_length, output_size)

        # RNN forward pass
        rnn_output, _ = self.rnn(input_sequence)

        # Fully connected layer for output
        output = self.fc(rnn_output)

        return output


full_adder = RNN(input_size=2, hidden_size=params[constants.HIDSIZE], output_size=1).to(
    device
)
adder_trn = BinaryAddData(
    dim=params[constants.NUM_BITS], num=params[constants.NUM_TRN], msb_first=False
)
adder_tst = BinaryAddData(
    dim=params[constants.NUM_BITS], num=params[constants.NUM_TST], msb_first=False
)
trn_loader = torch.utils.data.DataLoader(
    adder_trn, batch_size=params[constants.BSZ], shuffle=True
)
tst_loader = torch.utils.data.DataLoader(
    adder_tst, batch_size=params[constants.NUM_TST], shuffle=False
)
optim = torch.optim.Adam(full_adder.parameters(), lr=params[constants.LR])

logger.info(f"params: {params}")

for epoch in range(params[constants.NUM_EPOCHS]):
    for x, y in trn_loader:
        x = x.to(dtype=torch.float32, device=device)
        y = y.to(dtype=torch.float32, device=device)
        optim.zero_grad()
        output_sequence = full_adder(x)
        loss = nn.MSELoss()(output_sequence.squeeze(), y.squeeze())
        loss.backward()
        optim.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{params["num_epochs"]}], Loss: {loss.item():.4f}')
        logger.info(
            f'Epoch [{epoch+1}/{params["num_epochs"]}], Loss: {loss.item():.4f}'
        )

# Print the weights
logger.info(f"RNN Parameters")
for name, parameter in full_adder.rnn.named_parameters():
    logger.info(f"{name}: {parameter[0].cpu()}")
logger.info(f"H -> out parameters")
for name, parameter in full_adder.fc.named_parameters():
    logger.info(f"{name}: {parameter[0].cpu()}")

with torch.no_grad():
    for x, y in tst_loader:
        x = x.to(dtype=torch.float32, device=device)
        y = y.to(dtype=torch.float32, device=device)
        output_sequence = full_adder(x)
        loss = nn.MSELoss()(output_sequence, y)
        print(f"Test loss: {loss.item():.4f}")
        logger.info(f"Test loss: {loss.item():.4f}")
