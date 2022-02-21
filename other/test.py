from torch import nn
from transformers import BertModel


class CustomBERTModel(nn.Module):
    def __init__(self):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained("./src/bert-fa-base-uncased")
        ### New layers:
        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 2)  ## 3 is the number of classes in this example

    def forward(self, ids, mask):
        sequence_output, pooled_output = self.bert(
            ids,
            attention_mask=mask)

        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        linear1_output = self.linear1(sequence_output[:, 0, :].view(-1, 768))  ## extract the 1st token's embeddings

        linear2_output = self.linear2(linear1_output)

        return linear2_output


tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
model = CustomBERTModel()  # You can pass the parameters if required to have more flexible model
model.to(torch.device("cpu"))  ## can be gpu
criterion = nn.CrossEntropyLoss()  ## If required define your own criterion
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

for epoch in epochs:
    for batch in data_loader:  ## If you have a DataLoader()  object to get the data.

        data = batch[0]
        targets = batch[1]  ## assuming that data loader returns a tuple of data and its targets

        optimizer.zero_grad()
        encoding = tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True, truncation=True, max_length=50,
                                               add_special_tokens=True)
        outputs = model(input_ids, attention_mask=attention_mask)
        outputs = F.log_softmax(outputs, dim=1)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()