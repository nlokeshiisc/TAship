import torch

torch.manual_seed(0)
import torch.nn as nn
from dataset import BinaryAddData, BinaryGTData
import constants
import logging

task = "add"

log_file = (
    "binary_gt.log" if task == "gt" else "binary_add.log" if task == "add" else None
)
logging.basicConfig(
    filename=log_file,
    format="%(asctime)s :: %(filename)s:%(funcName)s :: %(message)s",
    filemode="a",
)
logger = logging.getLogger(name="binary_ops")
logger.setLevel(logging.DEBUG)

logger.info("****" * 20)
print("")

device = "cuda:6"

params = {
    constants.BSZ: 16,
    constants.NUM_BITS: 10,
    constants.NUM_TRN: 100,  # 'num_samples
    constants.NUM_TST: 100,
    constants.HIDSIZE: 4,
    constants.LR: 0.001,
    constants.NUM_EPOCHS: 3000,
    constants.STEP_ACT: False,
    constants.TRAIN_MODEL: True,
}


class StepAct(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input):
        return torch.relu(torch.sign(input))


class StepAct_2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input):
        scalar = 1.0
        return torch.sigmoid(scalar * torch.sign(input))


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        # Define an RNN layer
        self.rnn = nn.RNN(
            input_size=input_size, hidden_size=hidden_size, batch_first=True, nonlinearity="relu"
        )

        # Define a fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)
        self.step = StepAct()
        # self.step = StepAct_2()

    def forward(self, input_sequence):
        # Input sequence shape: (batch_size, sequence_length, input_size)
        # Output shape: (batch_size, sequence_length, output_size)

        # RNN forward pass
        rnn_output, _ = self.rnn(input_sequence)

        # Fully connected layer for output
        output = self.fc(rnn_output)

        if params[constants.STEP_ACT]:
            output = self.step(output)

        return output


model = RNN(input_size=2, hidden_size=params[constants.HIDSIZE], output_size=1).to(
    device
)
if task == "add":
    data_trn = BinaryAddData(
        dim=params[constants.NUM_BITS], num=params[constants.NUM_TRN], msb_first=False
    )
    data_tst = BinaryAddData(
        dim=params[constants.NUM_BITS], num=params[constants.NUM_TST], msb_first=False
    )
elif task == "gt":
    data_trn = BinaryGTData(
        dim=params[constants.NUM_BITS], num=params[constants.NUM_TRN], msb_first=True
    )
    data_tst = BinaryGTData(
        dim=params[constants.NUM_BITS], num=params[constants.NUM_TST], msb_first=True
    )

trn_loader = torch.utils.data.DataLoader(
    data_trn, batch_size=params[constants.BSZ], shuffle=True
)
tst_loader = torch.utils.data.DataLoader(
    data_tst, batch_size=params[constants.NUM_TST], shuffle=False
)
optim = torch.optim.Adam(model.parameters(), lr=params[constants.LR])

logger.info(f"params: {params}")

if params[constants.TRAIN_MODEL]:
    model.train()
    for epoch in range(params[constants.NUM_EPOCHS]):
        for x, y in trn_loader:
            x = x.to(dtype=torch.float32, device=device)
            y = y.to(dtype=torch.float32, device=device)
            optim.zero_grad()
            output_sequence = model(x)
            # select the last output of the sequence if task is gt
            if task == "gt":
                output_sequence = output_sequence[:, -1, :]
            loss = nn.MSELoss()(output_sequence.squeeze(), y.squeeze())
            loss.backward()
            optim.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{params["num_epochs"]}], Loss: {loss.item():.4f}')
            logger.info(
                f'Epoch [{epoch+1}/{params["num_epochs"]}], Loss: {loss.item():.4f}'
            )
    torch.save(model.state_dict(), "rnn_model.pth")
else:
    model.load_state_dict(torch.load("rnn_model.pth"))

model.eval()

with torch.no_grad():
    for x, y in tst_loader:
        x = x.to(dtype=torch.float32, device=device)
        y = y.to(dtype=torch.float32, device=device)
        output_sequence = model(x)
        if task == "gt":
            output_sequence = output_sequence[:, -1, :]
        loss = nn.MSELoss()(output_sequence.squeeze(), y.squeeze())
        print(f"Test loss: {loss.item():.4f}")
        logger.info(f"Test loss: {loss.item():.4f}")
        
        ydecimal = [data_tst.binary_to_decimal(y[i]) for i in range(y.size(0))]
        ypreddecimal = [data_tst.binary_to_decimal(output_sequence[i]) for i in range(output_sequence.size(0))]
        loss = nn.MSELoss()(torch.tensor(ydecimal), torch.tensor(ypreddecimal)) 
